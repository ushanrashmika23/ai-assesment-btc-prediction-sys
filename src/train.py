"""
Headless training entry point.

The script equivalent of `notebooks/05_local_training.ipynb`: same data, same
models, same metrics, but runnable from a terminal and repeatable. This is what
produces the `.keras` files in `models/` that `src/dashboard.py` and
`src/inference.py` load.

    # how long is one epoch on THIS machine? (trains one epoch, saves nothing)
    python src/train.py --probe

    # a quick end-to-end run on a tail subset of the training split
    python src/train.py --arch gru --subset 60000 --epochs 15

    # the full reference configuration
    python src/train.py --arch both --epochs 25

WHAT THIS DOES NOT DO
    It makes no accuracy, profitability or trading-edge claim. Every table it
    prints is a measurement on one historical test period, and every one of
    them is printed next to the naive baseline it has to beat. Directional
    accuracy is not profitability - there is no backtest here.

LEAKAGE DISCIPLINE, INHERITED FROM THE PIPELINE
    * Windows are read through `starts_*.npy`, never `X[k : k+lookback]`,
      because gap-excluded windows leave holes in the offset sequence.
    * The test split is touched exactly once, by the winner, at the end.
    * Architecture selection uses validation macro-F1 only.
    * The scalers and thresholds are LOADED from data/final/, never refitted.
    * `--subset` takes the most recent N training samples, never a random N.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import MODELS_DIR                                    # noqa: E402
from config import (                                             # noqa: E402
    DIRECTION_CLASS_WEIGHT_STRENGTH,
    ENSEMBLE_SEEDS,
    REGIME_CLASSES,
    REGIME_ENABLED,
    REGIME_HORIZONS,
    SELECTION_DIRECTION_WEIGHT,
    SELECTION_REGIME_WEIGHT,
)
from sequences import SequenceBatcher                            # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "final"

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]
REGIME_NAMES = [REGIME_CLASSES[i] for i in sorted(REGIME_CLASSES)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Artefacts:
    """Everything data/final/ ships, loaded and structurally checked."""

    def __init__(self, data_dir: Path = DATA_DIR):
        required = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "starts_train.npy", "starts_val.npy", "starts_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
            "y_dir_train.npy", "y_dir_val.npy", "y_dir_test.npy",
            "timestamps_train.npy", "timestamps_val.npy", "timestamps_test.npy",
            "scaler.pkl", "target_scaler.pkl", "dataset_metadata.json",
        ]
        if REGIME_ENABLED:
            required += ["y_regime_train.npy", "y_regime_val.npy",
                         "y_regime_test.npy"]
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {data_dir}: {', '.join(missing)}\n"
                f"Run `python src/pipeline.py` to build them."
            )

        for name in ("train", "val", "test"):
            setattr(self, f"X_{name}", np.load(data_dir / f"X_{name}.npy"))
            setattr(self, f"s_{name}", np.load(data_dir / f"starts_{name}.npy"))
            setattr(self, f"y_{name}", np.load(data_dir / f"y_{name}.npy"))
            setattr(self, f"d_{name}",
                    np.load(data_dir / f"y_dir_{name}.npy").astype(np.int64))
            setattr(self, f"ts_{name}",
                    np.load(data_dir / f"timestamps_{name}.npy"))
            if REGIME_ENABLED:
                setattr(self, f"r_{name}", np.load(
                    data_dir / f"y_regime_{name}.npy").astype(np.int64))

        with open(data_dir / "scaler.pkl", "rb") as fh:
            self.scaler = pickle.load(fh)
        with open(data_dir / "target_scaler.pkl", "rb") as fh:
            self.target_scaler = pickle.load(fh)
        with open(data_dir / "dataset_metadata.json", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        self.lookback = int(self.meta["lookback"])
        self.horizons = list(self.meta["horizons"].keys())        # labels
        self.horizon_candles = [int(self.meta["horizons"][h]["candles"])
                                for h in self.horizons]
        self.horizon_max = max(self.horizon_candles)
        self.n_h = len(self.horizons)
        self.n_f = int(self.X_train.shape[1])
        # Regime horizons, taken from the metadata block pipeline.py wrote when
        # it built y_regime_*.npy. Its `horizons` list IS the column order of
        # those arrays, so reading it here (rather than re-deriving it) is what
        # keeps the model's head order and the label columns in step.
        self.regime_meta = self.meta.get("regime_targets", {}) \
            if REGIME_ENABLED else {}
        self.regime_horizons = list(self.regime_meta.get("horizons", []))
        self.regime_horizon_candles = list(
            self.regime_meta.get("horizon_candles", []))
        self.n_regime = len(self.regime_horizons)

        self._check()

    def _check(self) -> None:
        """Block/offset misalignment would be silently fatal - assert it here."""
        for name in ("train", "val", "test"):
            X = getattr(self, f"X_{name}")
            s = getattr(self, f"s_{name}")
            y = getattr(self, f"y_{name}")
            d = getattr(self, f"d_{name}")
            assert len(s) == len(y) == len(d), f"{name}: offsets/targets misaligned"
            assert s.min() >= 0, f"{name}: negative window offset"
            assert s.max() + self.lookback <= len(X), f"{name}: window past block end"
            assert y.shape[1] == self.n_h, f"{name}: wrong horizon count"
            assert d.shape[1] == self.n_h, f"{name}: wrong horizon count"
            assert np.isfinite(X).all(), f"{name}: block contains NaN/Inf"
            if REGIME_ENABLED:
                r = getattr(self, f"r_{name}")
                assert self.n_regime == len(REGIME_HORIZONS), \
                    f"metadata lists {self.n_regime} regime horizons, config " \
                    f"asks for {len(REGIME_HORIZONS)}: {REGIME_HORIZONS}"
                assert self.regime_horizon_candles == list(REGIME_HORIZONS), \
                    f"regime columns are {self.regime_horizon_candles}, " \
                    f"config expects {list(REGIME_HORIZONS)}"
                assert r.shape[1] == self.n_regime, \
                    f"{name}: wrong regime horizon count"
                assert r.min() >= 0, f"{name}: negative regime label"
                assert len(r) == len(d), f"{name}: regime/targets misaligned"

    def thresholds(self) -> dict[str, tuple[float, float]]:
        per = self.meta["thresholds"]["per_horizon"]
        return {h: (per[h]["lower_percent"], per[h]["upper_percent"])
                for h in self.horizons}

    def to_scaled(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.target_scaler.mean_.astype(np.float32))
                / self.target_scaler.scale_.astype(np.float32)).astype(np.float32)

    def from_scaled(self, y: np.ndarray) -> np.ndarray:
        return (y * self.target_scaler.scale_.astype(np.float32)
                + self.target_scaler.mean_.astype(np.float32)).astype(np.float32)

    def purge_clearance(self) -> int:
        """
        Candles of margin between the last train label and the first
        validation feature candle.

        Uses TIMESTAMPS, not starts_*.npy: those hold block-relative offsets,
        so starts_val[0] is 0 and comparing splits through them is meaningless.
        Target timestamps are absolute UTC and directly comparable.
        """
        candle_ms = 15 * 60 * 1000
        gap = (int(self.ts_val[0]) - int(self.ts_train[-1])) // candle_ms
        required = self.lookback + self.horizon_max
        if gap < required:
            raise AssertionError(
                f"Training labels overlap the validation feature window: splits "
                f"are {gap} candles apart but {required} are needed. Re-run "
                f"src/pipeline.py and its audit."
            )
        return gap - required


def subset_train(art: Artefacts, n_keep: int):
    """
    Most recent `n_keep` training samples, as a self-consistent block+offsets.

    Chronological, not random: the subset stays contiguous and adjacent to the
    validation split. It is still strictly before the purge boundary, so no
    choice here can touch the test split.
    """
    n_total = len(art.y_train)
    if n_keep is None or n_keep >= n_total:
        return (art.X_train, art.s_train, art.y_train, art.d_train,
                art.r_train if REGIME_ENABLED else None)

    cut = n_total - int(n_keep)
    starts = art.s_train[cut:]
    lo = int(starts[0])
    return (art.X_train[lo:], (starts - lo).astype(np.int64),
            art.y_train[cut:], art.d_train[cut:],
            art.r_train[cut:] if REGIME_ENABLED else None)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def make_sequence(keras, X_block, y_scaled, y_dir, starts, lookback,
                  batch_size, shuffle=False, seed=None, y_regime=None):
    """
    A keras.utils.PyDataset streaming (window, {return_head, direction_head,
    regime_head}).

    PyDataset is the supported Keras 3 way to stream a Python sequence. The
    windowing itself is src/sequences.py::SequenceBatcher - the project's own
    code - so training and inference cannot drift apart.
    """

    class WindowSequence(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.core = SequenceBatcher(X_block, y_scaled, starts,
                                        lookback=lookback,
                                        batch_size=batch_size, shuffle=False)
            self.ydir = np.asarray(y_dir, dtype=np.int64)
            self.yreg = (None if y_regime is None
                         else np.asarray(y_regime, dtype=np.int64))
            self.n = int(self.core.n_samples)
            self.shuffle = bool(shuffle)
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(self.n, dtype=np.int64)

        def __len__(self):
            return int(np.ceil(self.n / self.core.batch_size))

        def __getitem__(self, i):
            chunk = self.order[i * self.core.batch_size:
                               (i + 1) * self.core.batch_size]
            Xb, yb = self.core.materialise(chunk)
            out = {"return_head": yb, "direction_head": self.ydir[chunk]}
            if self.yreg is not None:
                out["regime_head"] = self.yreg[chunk]
            return Xb, out

        def on_epoch_end(self):
            if self.shuffle:
                self.order = self.rng.permutation(self.n)

    return WindowSequence()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def make_weighted_scc(keras, class_weights):
    """
    Sparse categorical cross-entropy with a per-class weight vector.

    `class_weights` is (n_horizons, n_classes) - one weight per (horizon, class)
    - so the direction head can be stopped from collapsing onto NEUTRAL without
    touching the regime head. Weights are applied to the per-sample loss and the
    mean is taken afterwards, which keeps the loss scale independent of the
    batch's class mix.
    """
    import tensorflow as tf

    w = tf.constant(np.asarray(class_weights, dtype=np.float32))

    class WeightedSCC(keras.losses.Loss):
        def __init__(self, name="weighted_scc"):
            super().__init__(name=name)

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            n_cls = tf.shape(w)[1]
            onehot = tf.one_hot(y_true, n_cls, dtype=tf.float32)
            per_class = tf.reduce_sum(onehot * w[None, :, :], axis=-1)
            logp = tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
            ce = -tf.reduce_sum(onehot * logp, axis=-1)
            return tf.reduce_mean(per_class * ce)

        def get_config(self):
            return super().get_config()

    return WeightedSCC()


def compute_direction_class_weights(y_dir_train, n_h, n_classes=3,
                                    strength=DIRECTION_CLASS_WEIGHT_STRENGTH):
    """
    (n_horizons, n_classes) weights, from TRAIN labels only.

    w_c is proportional to (1 / frequency_c) ** strength, then normalised so the
    mean weight is 1. strength=0 gives a uniform vector (identical to plain
    cross-entropy); strength=1 gives full inverse-frequency weighting.
    """
    weights = np.ones((n_h, n_classes), dtype=np.float64)
    if strength <= 0:
        return weights
    for i in range(n_h):
        col = y_dir_train[:, i]
        col = col[col >= 0]
        if col.size == 0:
            continue
        freq = np.bincount(col, minlength=n_classes).astype(np.float64)
        freq = np.maximum(freq, 1.0)
        freq = freq / freq.sum()
        w = (1.0 / freq) ** float(strength)
        weights[i] = w / w.mean()
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_multi_task_model(keras, cell_type, units, dropout, learning_rate,
                           n_h, n_f, lookback, n_regime=0,
                           direction_class_weights=None, return_loss_weight=1.0,
                           regime_loss_weight=1.0):
    """
    Shared recurrent trunk, up to three heads.

    The multi-task setup is the point: the direction head pushes the trunk
    toward representations that separate the three directional classes, the
    regime head pushes it toward representations of upcoming ACTIVITY, and the
    regression head keeps the magnitude information the class labels discard.
    Regressing on SCALED returns, because raw percent returns are ~0.39% at
    15m and their MSE gradients are tiny.

    The regime head is what makes the model's accuracy headline defensible: it
    answers "will the next window be active or quiet?", a question this dataset
    carries real signal for, and its label is stationary across splits. See
    src/regime.py for the target and src/config.py for why.

    Heads are ordered (return, direction, regime). The softmax/reshape ordering
    argument below applies to BOTH classification heads.
    """
    layer = keras.layers.GRU if cell_type.upper() == "GRU" else keras.layers.LSTM

    inputs = keras.Input(shape=(lookback, n_f), name="window")
    x = layer(units, return_sequences=True, name=f"{cell_type.lower()}_1")(inputs)
    x = keras.layers.Dropout(dropout, name="drop_1")(x)
    x = layer(units, return_sequences=False, name=f"{cell_type.lower()}_2")(x)
    x = keras.layers.Dropout(dropout, name="drop_2")(x)
    x = keras.layers.LayerNormalization(name="trunk_norm")(x)

    return_head = keras.layers.Dense(n_h, name="return_head")(x)

    # ORDER MATTERS. Reshape to (n_h, 3) FIRST, then softmax over the class
    # axis. Applying softmax to the flat 18-vector and reshaping afterwards
    # shares ONE normaliser across all six horizons, so each horizon's three
    # probabilities sum to ~1/6 instead of 1: the head cannot express "UP at
    # 15m, DOWN at 90m" independently, and the confidence it reports is not a
    # probability. Argmax still happens to survive (softmax is monotonic within
    # a group), which is exactly why this is easy to miss - the self-check at
    # the end of this function asserts the normalisation.
    d = keras.layers.Dense(n_h * 3, name="direction_logits")(x)
    d = keras.layers.Reshape((n_h, 3), name="direction_reshape")(d)
    direction_head = keras.layers.Activation("softmax",
                                             name="direction_head")(d)

    outputs = {"return_head": return_head, "direction_head": direction_head}
    losses = {"return_head": "mse",
              "direction_head": (make_weighted_scc(keras,
                                                   direction_class_weights)
                                 if direction_class_weights is not None
                                 else keras.losses.SparseCategoricalCrossentropy())}
    loss_weights = {"return_head": return_loss_weight,
                    "direction_head": 1.0}
    metrics = {"return_head": ["mae"],
               "direction_head": ["sparse_categorical_accuracy"]}

    if n_regime:
        r = keras.layers.Dense(n_regime * 2, name="regime_logits")(x)
        r = keras.layers.Reshape((n_regime, 2), name="regime_reshape")(r)
        regime_head = keras.layers.Activation("softmax",
                                              name="regime_head")(r)
        outputs["regime_head"] = regime_head
        losses["regime_head"] = keras.losses.SparseCategoricalCrossentropy()
        loss_weights["regime_head"] = regime_loss_weight
        metrics["regime_head"] = ["sparse_categorical_accuracy"]

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{cell_type.lower()}_multitask",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # Cheap structural self-check. Getting the softmax/reshape order wrong is
    # silent - it still trains, still produces an argmax, and only shows up as
    # probabilities that do not sum to 1. Assert it here rather than discover
    # it after an hour of training.
    probe = model.predict(np.zeros((2, lookback, n_f), dtype=np.float32),
                          verbose=0)
    for head, n_cls in (("direction_head", 3), ("regime_head", 2)):
        if head not in probe:
            continue
        if not np.allclose(probe[head].sum(axis=-1), 1.0, atol=1e-5):
            raise AssertionError(
                f"{head} probabilities do not sum to 1 per horizon - the "
                f"softmax is almost certainly applied before the reshape"
            )
        if probe[head].shape[-1] != n_cls:
            raise AssertionError(f"{head}: expected {n_cls} classes")
    return model


# ---------------------------------------------------------------------------
# Validation-metric callback
# ---------------------------------------------------------------------------

class SelectionScoreCallback:
    """
    Scores the validation split at the end of every epoch and injects the
    result into `logs`, so the Keras callbacks AFTER it in the list can monitor
    a metric they could not otherwise see.

    Keras offers no built-in macro-F1, and the loss it would otherwise monitor
    (`val_loss`) is dominated by the return head, which this project has already
    measured to have no skill at all (test correlation ~0.02, worse than
    predicting the mean at every horizon). Early-stopping on it was selecting
    for the one head that cannot predict anything.

    What it computes:

        direction_macro_F1   macro-F1 over DOWN/NEUTRAL/UP, averaged over the
                             six direction horizons
        regime_accuracy      accuracy over QUIET/ACTIVE, averaged over the
                             regime horizons
        selection_score      SELECTION_DIRECTION_WEIGHT * direction_macro_F1
                             + SELECTION_REGIME_WEIGHT * regime_accuracy

    Must be constructed with a factory that receives `keras` (module-level
    TensorFlow stays inside main()).
    """

    def __new__(cls, keras, val_seq, y_dir_val, y_regime_val, horizons,
                regime_horizons):
        from sklearn.metrics import accuracy_score, f1_score

        class _Cb(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs if logs is not None else {}
                raw = self.model.predict(val_seq, verbose=0)
                d_pred = np.asarray(raw["direction_head"]).argmax(axis=-1)

                f1s = [f1_score(y_dir_val[:, i], d_pred[:, i],
                                average="macro", zero_division=0)
                       for i in range(len(horizons))]
                direction_f1 = float(np.mean(f1s))
                logs["val_direction_macro_f1"] = direction_f1
                logs["val_direction_accuracy"] = float(np.mean(
                    [accuracy_score(y_dir_val[:, i], d_pred[:, i])
                     for i in range(len(horizons))]))

                if "regime_head" in raw and y_regime_val is not None:
                    r_pred = np.asarray(raw["regime_head"]).argmax(axis=-1)
                    accs = [accuracy_score(y_regime_val[:, i], r_pred[:, i])
                            for i in range(len(regime_horizons))]
                    logs["val_regime_accuracy"] = float(np.mean(accs))
                    logs["val_regime_macro_f1"] = float(np.mean(
                        [f1_score(y_regime_val[:, i], r_pred[:, i],
                                  average="macro", zero_division=0)
                         for i in range(len(regime_horizons))]))
                else:
                    logs["val_regime_accuracy"] = 0.0
                    logs["val_regime_macro_f1"] = 0.0

                logs["val_selection_score"] = (
                    SELECTION_DIRECTION_WEIGHT * direction_f1
                    + SELECTION_REGIME_WEIGHT * logs["val_regime_accuracy"]
                )

        return _Cb()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_corr(a, b) -> float:
    """Pearson r, or NaN when either side is constant (the naive baseline)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y_true, y_pred, horizons) -> pd.DataFrame:
    rows = {}
    for i, h in enumerate(horizons):
        e = y_pred[:, i] - y_true[:, i]
        rows[h] = {
            "MAE_%": float(np.mean(np.abs(e))),
            "RMSE_%": float(np.sqrt(np.mean(e ** 2))),
            "bias_%": float(np.mean(e)),
            "corr": _safe_corr(y_true[:, i], y_pred[:, i]),
        }
    return pd.DataFrame(rows).T


def direction_metrics(y_true_dir, y_pred_dir, horizons) -> pd.DataFrame:
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    rows = {}
    for i, h in enumerate(horizons):
        t, p = y_true_dir[:, i], y_pred_dir[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(
            t, p, labels=[0, 1, 2], zero_division=0)
        rows[h] = {
            "accuracy": accuracy_score(t, p),
            "macro_F1": f1_score(t, p, average="macro", zero_division=0),
            "DOWN_prec": prec[0], "DOWN_rec": rec[0], "DOWN_f1": f1[0],
            "NEUTRAL_prec": prec[1], "NEUTRAL_rec": rec[1], "NEUTRAL_f1": f1[1],
            "UP_prec": prec[2], "UP_rec": rec[2], "UP_f1": f1[2],
            "n": int(len(t)),
        }
    return pd.DataFrame(rows).T


def regime_metrics(y_true_reg, y_pred_reg, horizons) -> pd.DataFrame:
    """
    Accuracy / macro-F1 / per-class detail for the QUIET-ACTIVE head.

    Unlike the direction table, accuracy here is the honest headline: the two
    classes are ~50/50 by construction (the threshold is the TRAIN median of a
    self-normalising ratio), so there is no majority class to hide behind.
    """
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    rows = {}
    for i, h in enumerate(horizons):
        t, p = y_true_reg[:, i], y_pred_reg[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(
            t, p, labels=[0, 1], zero_division=0)
        rows[h] = {
            "accuracy": accuracy_score(t, p),
            "macro_F1": f1_score(t, p, average="macro", zero_division=0),
            "base_rate": float(max(np.bincount(t, minlength=2)) / len(t)),
            "QUIET_prec": prec[0], "QUIET_rec": rec[0], "QUIET_f1": f1[0],
            "ACTIVE_prec": prec[1], "ACTIVE_rec": rec[1], "ACTIVE_f1": f1[1],
            "n": int(len(t)),
        }
    return pd.DataFrame(rows).T


def naive_regime(y_shape, train_reg) -> np.ndarray:
    """Always predict the majority TRAINING regime class, per horizon."""
    preds = np.zeros(y_shape, dtype=np.int64)
    for i in range(y_shape[1]):
        vals, counts = np.unique(train_reg[:, i], return_counts=True)
        preds[:, i] = vals[np.argmax(counts)]
    return preds


def naive_direction(y_shape, train_dir) -> np.ndarray:
    """Always predict the majority TRAINING class, per horizon."""
    preds = np.zeros(y_shape, dtype=np.int64)
    for i in range(y_shape[1]):
        vals, counts = np.unique(train_dir[:, i], return_counts=True)
        preds[:, i] = vals[np.argmax(counts)]
    return preds


def bar(value: float, width: int = 22) -> str:
    """A crude ASCII bar, for eyeballing a 0-1 metric."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the multi-task GRU/LSTM on the prepared dataset.")
    p.add_argument("--arch", default="gru", choices=["gru", "lstm", "both"])
    p.add_argument("--subset", type=int, default=None,
                   help="most recent N training samples (default: all)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds to train and average into an "
                        "ensemble, e.g. 42,43,44. Default: just --seed.")
    p.add_argument("--ensemble", default=None,
                   help="comma-separated SEEDS to mix into the reported "
                        "ensemble (default: every seed trained). Use to drop a "
                        "seed that trained badly without retraining.")
    p.add_argument("--no-ensemble", action="store_true",
                   help="report the single best seed instead of averaging")
    p.add_argument("--threads", type=int, default=0,
                   help="intra-op threads (default: cores - 2)")
    p.add_argument("--probe", action="store_true",
                   help="time ONE epoch then exit without saving anything")
    p.add_argument("--tag", default=None,
                   help="output name (default: <arch>); use to avoid clobbering")
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable direction class weighting (plain cross-entropy)")
    p.add_argument("--class-weight-strength", type=float,
                   default=DIRECTION_CLASS_WEIGHT_STRENGTH,
                   help="0 = uniform, 1 = full inverse-frequency")
    p.add_argument("--return-loss-weight", type=float, default=1.0,
                   help="weight on the return head's MSE (it has no skill; "
                        "lowering it stops it dominating the trunk)")
    args = p.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    import tensorflow as tf
    from tensorflow import keras

    print(f"python {sys.version.split()[0]}   tensorflow {tf.__version__}   "
          f"keras {keras.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"device: {'GPU ' + str(gpus) if gpus else 'CPU only'}")

    n_threads = args.threads or max(1, (os.cpu_count() or 4) - 2)
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, n_threads // 2))
    print(f"threads: {n_threads} intra-op, {max(1, n_threads // 2)} inter-op")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    # -- data ---------------------------------------------------------------
    art = Artefacts()
    clearance = art.purge_clearance()
    print(f"\nlookback {art.lookback}   features {art.n_f}   "
          f"horizons {art.horizons}   HORIZON_MAX {art.horizon_max} candles")
    print(f"purge clearance: {clearance} candle(s)")

    X_fit, s_fit, y_fit, d_fit, r_fit = subset_train(art, args.subset)
    if len(y_fit) < len(art.y_train):
        print(f"using the most recent {len(y_fit):,} of {len(art.y_train):,} "
              f"training samples  <-- SUBSET, not a reportable configuration")
    else:
        print(f"using the full training split: {len(y_fit):,} samples")

    print(f"val {len(art.y_val):,}   test {len(art.y_test):,}")

    n_regime = art.n_regime
    if n_regime:
        print(f"regime head: {art.regime_horizons} "
              f"(QUIET/ACTIVE) alongside {art.n_h} direction horizons")
    else:
        print("regime head: DISABLED")

    fit_seq = make_sequence(keras, X_fit, art.to_scaled(y_fit), d_fit, s_fit,
                            art.lookback, args.batch, shuffle=True, seed=args.seed,
                            y_regime=r_fit)
    val_seq = make_sequence(keras, art.X_val, art.to_scaled(art.y_val), art.d_val,
                            art.s_val, art.lookback, args.batch,
                            y_regime=art.r_val if n_regime else None)
    test_seq = make_sequence(keras, art.X_test, art.to_scaled(art.y_test),
                             art.d_test, art.s_test, art.lookback, args.batch,
                             y_regime=art.r_test if n_regime else None)
    print(f"batches: {len(fit_seq)} fit | {len(val_seq)} val | {len(test_seq)} test")

    # Direction class weights, from TRAIN labels only.
    dir_weights = None
    if not args.no_class_weights and args.class_weight_strength > 0:
        dir_weights = compute_direction_class_weights(
            d_fit, art.n_h, strength=args.class_weight_strength)
        print(f"direction class weights (strength "
              f"{args.class_weight_strength:g}), per horizon DOWN/NEUTRAL/UP:")
        for i, h in enumerate(art.horizons):
            print(f"  {h:>4}  " + "  ".join(f"{w:.3f}" for w in dir_weights[i]))

    def build(cell, seed):
        return build_multi_task_model(
            keras, cell, args.units, args.dropout, args.lr,
            art.n_h, art.n_f, art.lookback, n_regime=n_regime,
            direction_class_weights=dir_weights,
            return_loss_weight=args.return_loss_weight)

    def callbacks_for(ckpt, seed):
        """
        Order matters. SelectionScoreCallback writes val_selection_score into
        `logs`, and the three callbacks after it monitor that key - so it must
        come first or they will never see it.
        """
        return [
            SelectionScoreCallback(keras, val_seq, art.d_val,
                                   art.r_val if n_regime else None,
                                   art.horizons, art.regime_horizons),
            keras.callbacks.EarlyStopping(monitor="val_selection_score",
                                          mode="max", patience=args.patience,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_selection_score",
                                              mode="max", factor=0.5, patience=2,
                                              min_lr=1e-5, verbose=1),
            keras.callbacks.ModelCheckpoint(ckpt, monitor="val_selection_score",
                                            mode="max", save_best_only=True,
                                            verbose=0),
        ]

    # -- probe: how long is one epoch here? ---------------------------------
    if args.probe:
        print("\nPROBE - one epoch on the data above, nothing is saved.\n")
        probe = build("GRU", args.seed)
        t0 = time.time()
        probe.fit(fit_seq, validation_data=val_seq, epochs=1, verbose=2)
        dt = time.time() - t0
        steps = len(fit_seq)
        print(f"\none epoch: {dt:.1f}s  ({steps} steps, "
              f"{dt / max(1, steps) * 1000:.0f} ms/step)")
        print(f"  + ~{len(val_seq)} val batches of callback overhead per epoch")
        for n in (5, 10, 25, 50):
            print(f"  {n:>3} epochs -> {dt * n / 60:6.1f} min")
        print(f"\nfull split: {len(art.y_train):,} samples")
        return 0

    # -- train --------------------------------------------------------------
    to_run = ["gru", "lstm"] if args.arch == "both" else [args.arch]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results, histories, timings = {}, {}, {}
    members = {}          # (cell, seed) -> path to the saved best weights

    for cell in to_run:
        for seed in seeds:
            tag = args.tag or cell
            name = tag if len(seeds) == 1 else f"{tag}_s{seed}"
            print(f"\n{'=' * 72}\nTRAINING {cell.upper()}  ({name})  "
                  f"seed {seed}\n{'=' * 72}")
            keras.utils.set_random_seed(seed)
            model = build(cell, seed)
            ckpt = MODELS_DIR / f"{name}_best.keras"
            t0 = time.time()
            hist = model.fit(fit_seq, validation_data=val_seq, epochs=args.epochs,
                             callbacks=callbacks_for(ckpt, seed), verbose=2)
            elapsed = time.time() - t0
            n_ep = len(hist.history["loss"])
            print(f"\n{name}: {n_ep} epoch(s) in {elapsed / 60:.1f} min "
                  f"({elapsed / max(1, n_ep):.0f}s/epoch) -> {ckpt.name}")

            raw = model.predict(val_seq, verbose=0)
            val_dir = raw["direction_head"].argmax(axis=-1).astype(np.int64)
            results[name] = direction_metrics(art.d_val, val_dir, art.horizons)
            if n_regime:
                results[name + "_regime"] = regime_metrics(
                    art.r_val, raw["regime_head"].argmax(axis=-1),
                    art.regime_horizons)
            histories[name] = hist
            timings[name] = {"epochs": n_ep, "seconds": elapsed, "seed": seed}
            members[(cell, seed)] = ckpt
            del model

            # The regime head covers a SUBSET of the direction horizons, so the
            # two tables have different indexes - iterate both explicitly.
            print(f"\n{name} - VALIDATION direction macro-F1 / regime accuracy")
            for h in art.horizons:
                f1 = results[name].loc[h, "macro_F1"]
                rtxt = ""
                if h in art.regime_horizons:
                    rtxt = (f"   regime "
                            f"{results[name + '_regime'].loc[h, 'accuracy']:.4f}")
                print(f"  {h:>4}  dir {f1:.4f}  {bar(f1)}{rtxt}")

    # -- ensemble the seeds, select on validation, then touch test once -----
    keep = ([int(s) for s in args.ensemble.split(",") if s.strip()]
            if args.ensemble else list(seeds))
    if args.no_ensemble:
        keep = []
    unknown = [s for s in keep if s not in seeds]
    if unknown:
        raise SystemExit(f"--ensemble seeds {unknown} were not trained "
                         f"(trained: {seeds})")

    def ensemble_predict(cell, seq):
        """
        Average the softmax probabilities of every kept seed for one cell.

        Averaging probabilities (not logits, not hard votes) is the standard
        choice: each member is a proper distribution, so the mean is too, and it
        lets a confident-but-outvoted member still contribute. Returns a dict
        with the same keys the single-model `predict` returned, so everything
        downstream is unchanged.
        """
        use = keep or list(seeds)
        acc = None
        for sd in use:
            # compile=False: the direction loss is a locally-defined
            # WeightedSCC that Keras cannot deserialize by name. Nothing
            # here trains, so the compiled loss is not wanted.
            m = keras.models.load_model(members[(cell, sd)], compile=False)
            out = m.predict(seq, verbose=0)
            if acc is None:
                acc = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
            else:
                for k, v in out.items():
                    acc[k] += np.asarray(v, dtype=np.float64)
            del m
        return {k: v / len(use) for k, v in acc.items()}

    # Validation score of each candidate, on the exact artefact that will be
    # reported. Ensembling happens BEFORE selection, so the thing chosen is the
    # thing measured.
    val_scores = {}
    for cell in to_run:
        raw = ensemble_predict(cell, val_seq)
        v_dir = direction_metrics(
            art.d_val, raw["direction_head"].argmax(-1).astype(np.int64),
            art.horizons)
        v_reg = (regime_metrics(art.r_val, raw["regime_head"].argmax(-1),
                                art.regime_horizons) if n_regime else None)
        # Must mirror SelectionScoreCallback exactly, or selection is off-policy.
        score = (SELECTION_DIRECTION_WEIGHT * float(v_dir["macro_F1"].mean())
                 + (SELECTION_REGIME_WEIGHT * float(v_reg["accuracy"].mean())
                    if n_regime else 0.0))
        if not n_regime:
            score = float(v_dir["macro_F1"].mean())
        val_scores[cell] = score
        n_members = len(keep or seeds)
        label = f"{cell} x{n_members}" if n_members > 1 else cell
        print(f"\nvalidation  {label:<12} "
              f"dir macro-F1 {float(v_dir['macro_F1'].mean()):.4f}"
              + (f"   regime acc {float(v_reg['accuracy'].mean()):.4f}"
                 if n_regime else "")
              + f"   selection score {score:.4f}")

    best = max(val_scores, key=val_scores.get)
    print(f"\nselected on validation: {best.upper()}")
    if len(val_scores) > 1:
        print("A small gap is a hypothesis, not a conclusion - re-run with "
              "other seeds before reading anything into it.")
    if args.no_ensemble and len(seeds) > 1:
        print("(--no-ensemble: reporting a single seed)")

    raw = ensemble_predict(best, test_seq)
    test_pred = art.from_scaled(raw["return_head"])
    test_dir = raw["direction_head"].argmax(axis=-1).astype(np.int64)

    naive_reg = regression_metrics(art.y_test, np.zeros_like(art.y_test),
                                   art.horizons)
    naive_dir = direction_metrics(
        art.d_test, naive_direction(art.d_test.shape, art.d_train), art.horizons)
    m_reg = regression_metrics(art.y_test, test_pred, art.horizons)
    m_dir = direction_metrics(art.d_test, test_dir, art.horizons)
    m_regime = m_naive_regime = None
    if n_regime:
        m_regime = regime_metrics(art.r_test, raw["regime_head"].argmax(-1),
                                  art.regime_horizons)
        m_naive_regime = regime_metrics(
            art.r_test, naive_regime(art.r_test.shape, art.r_train),
            art.regime_horizons)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    members_used = keep or list(seeds)
    label = (f"{best.upper()} x{len(members_used)}"
             if len(members_used) > 1 else best.upper())

    print(f"\n{'=' * 72}\nTEST - {label} vs naive baselines "
          f"({len(art.y_test):,} samples)\n{'=' * 72}")

    # The regime head is the accuracy headline. It is reported first, with its
    # own baseline, because it is the only head here whose accuracy is not
    # swamped by a majority class.
    if n_regime:
        print("\nREGIME accuracy (higher is better) - THE ACCURACY HEADLINE")
        print("  QUIET vs ACTIVE: will the next h candles be more volatile than")
        print("  the trailing 24h average? Both classes are ~50% of the test set.")
        print(pd.DataFrame({
            "majority_class": m_naive_regime["accuracy"],
            f"{best}_model": m_regime["accuracy"],
            "lift": m_regime["accuracy"] - m_naive_regime["accuracy"],
        }).to_string())
        print("\nregime macro-F1")
        print(pd.DataFrame({
            "majority_class": m_naive_regime["macro_F1"],
            f"{best}_model": m_regime["macro_F1"],
        }).to_string())
        print(f"\n{best} - regime per-class detail")
        print(m_regime[["accuracy", "macro_F1", "base_rate",
                        "QUIET_prec", "QUIET_rec", "ACTIVE_prec", "ACTIVE_rec",
                        "n"]].to_string())

    print("\nDIRECTION macro-F1 (higher is better) - the harder, weaker question")
    print(pd.DataFrame({"majority_class": naive_dir["macro_F1"],
                        f"{best}_model": m_dir["macro_F1"]}).to_string())
    print("\ndirection accuracy (shown for completeness; NEUTRAL dominates it)")
    print(pd.DataFrame({"majority_class": naive_dir["accuracy"],
                        f"{best}_model": m_dir["accuracy"]}).to_string())

    print(f"\n{best} - direction per-class detail")
    print(m_dir[["accuracy", "macro_F1", "DOWN_prec", "DOWN_rec",
                 "UP_prec", "UP_rec", "NEUTRAL_prec", "NEUTRAL_rec"]].to_string())

    print("\nMAE in percent return (lower is better)")
    print(pd.DataFrame({"naive_zero": naive_reg["MAE_%"],
                        f"{best}_model": m_reg["MAE_%"]}).to_string())

    # -- save ---------------------------------------------------------------
    # With an ensemble there is no single model object to save, so the members
    # stay as the per-seed checkpoints and the manifest below says how to
    # combine them. inference.py follows the same recipe.
    if len(members_used) > 1:
        final_path = None
        print(f"\nensemble of {len(members_used)} members "
              f"({', '.join(f'{best}_s{s}' for s in members_used)}) - "
              f"no single .keras is written")
    else:
        sd = members_used[0]
        member_path = members[(best, sd)]
        final_path = MODELS_DIR / f"{best}_multitask.keras"
        # Re-save from a compile=False load so the shipped file carries no
        # optimizer state and no custom-loss reference: it loads anywhere
        # with a plain keras.models.load_model().
        keras.models.load_model(member_path, compile=False).save(final_path)

    bundle = {
        "architecture": best.upper(),
        "trained_by": "src/train.py",
        "trained_utc": pd.Timestamp.now("UTC").isoformat(),
        "lookback": art.lookback,
        "feature_columns": art.meta["feature_columns"],
        "horizons": art.horizons,
        "horizon_candles": art.horizon_candles,
        "direction_classes": art.meta["direction_classes"],
        "thresholds": art.meta["thresholds"]["per_horizon"],
        "target_scaler_mean": np.asarray(art.target_scaler.mean_).tolist(),
        "target_scaler_scale": np.asarray(art.target_scaler.scale_).tolist(),
        "train_samples_used": int(len(y_fit)),
        "train_samples_available": int(len(art.y_train)),
        "subset_used": bool(len(y_fit) < len(art.y_train)),
        "seed": args.seed,
        # Copied verbatim from the dataset metadata so a model file can never
        # be paired with thresholds from a different pipeline run.
        "regime": art.regime_meta,
        "direction_class_weights": (None if dir_weights is None
                                    else np.asarray(dir_weights).tolist()),
        "ensemble": {
            "members": [{"architecture": best.upper(), "seed": int(s),
                         "path": members[(best, s)].name}
                        for s in members_used],
            "combine": "mean of per-member softmax probabilities",
        },
        "selection": {
            "metric": "val_selection_score",
            "formula": (f"{SELECTION_DIRECTION_WEIGHT:g} * mean val direction "
                        f"macro-F1 + {SELECTION_REGIME_WEIGHT:g} * mean val "
                        f"regime accuracy"),
            "validation_scores": val_scores,
        },
    }
    with open(MODELS_DIR / "inference_bundle.json", "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)

    metrics = {
        "config": vars(args),
        "batches": {"fit": len(fit_seq), "val": len(val_seq),
                    "test": len(test_seq)},
        "timings": timings,
        "selected_on_validation": best.upper(),
        "ensemble_members": members_used,
        "validation_scores": val_scores,
        "test_mae_naive_zero": naive_reg["MAE_%"].to_dict(),
        "test_macro_f1_majority": naive_dir["macro_F1"].to_dict(),
        "test_regression": m_reg.to_dict(),
        "test_direction": m_dir.to_dict(),
    }
    if n_regime:
        metrics["test_regime"] = m_regime.to_dict()
        metrics["test_regime_majority"] = m_naive_regime.to_dict()
    with open(MODELS_DIR / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nsaved to {MODELS_DIR}:")
    to_list = [MODELS_DIR / "inference_bundle.json",
               MODELS_DIR / "train_metrics.json"]
    if final_path is not None:
        to_list.insert(0, final_path)
    for f in to_list:
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    print("""
Before quoting any number above:
  1. REGIME accuracy is the headline. It has two ~50% classes, so its baseline
     is ~0.50 and the lift is real. Quote its lift, not the raw number.
  2. DIRECTION macro-F1 is the weak head. Check it still beats majority_class
     and that no class has collapsed (per-class recalls).
  3. Did the model beat naive_zero on MAE? It never has - say so plainly.
  4. Was the test set touched exactly once? Everything above was selected on
     validation; nothing in this script reads the test split before this block.

Regime accuracy is not profitability either. Predicting "this will be an active
window" is not a trade: there is no backtest, no fees, no slippage, and the test
period is one regime. No claim is made.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
