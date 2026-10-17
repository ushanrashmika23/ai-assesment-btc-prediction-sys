"""
Rolling sequence generation, chronological splitting and scaling.

Geometry
--------
    X : (n_samples, LOOKBACK, n_features)      LOOKBACK = 64 candles = 16h
    y : (n_samples, n_horizons)                six future percentage returns

Sample i is built from 2D rows [i, i + LOOKBACK - 1]; its target is the target
of the LAST row of that window (row i + LOOKBACK - 1). Sequences roll forward
one row at a time and are never shuffled at construction time.

Memory
------
The full 3D tensor for ~9 years of 15m candles is several GB, while the
equivalent 2D matrix is ~60 MB. So the 3D tensor is produced as a ZERO-COPY
sliding-window view (`make_sequences`) and only materialised per split, per
batch (`SequenceBatcher`). Nothing here duplicates the feature matrix.

Leakage controls
----------------
* A window may not straddle a data gap (segment change) - enforced analytically
  with a cumulative sum, O(n) time and O(n) bytes.
* Splits are strictly chronological: train -> validation -> test.
* A purge buffer of PURGE_SAMPLES samples is discarded at each split boundary
  so the last training target cannot overlap the first validation feature
  window.
* The scaler is fitted on the TRAIN split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Zero-copy windowing
# ----------------------------------------------------------------------------

def make_sequences(X: np.ndarray, lookback: int) -> np.ndarray:
    """
    (n_rows, n_features) -> (n_rows - lookback + 1, lookback, n_features) view.

    This is a VIEW, not a copy: it costs no memory. Materialise a batch with
    np.ascontiguousarray(...) before handing it to a model, or use
    SequenceBatcher which does that for you.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X.shape}")
    if len(X) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(X)}")
    sw = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)
    # (n_windows, n_features, lookback) -> (n_windows, lookback, n_features)
    return sw.transpose(0, 2, 1)


def valid_window_starts(
    segment_ids: np.ndarray,
    target_valid: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Indices i such that:
      * 2D rows i .. i+lookback-1 all belong to one contiguous segment, and
      * the target row (i + lookback - 1) has a usable target.

    The segment test uses a prefix sum of "segment changed here" flags, so no
    (n_windows x lookback) boolean matrix is ever allocated.
    """
    segment_ids = np.asarray(segment_ids)
    target_valid = np.asarray(target_valid, dtype=bool)
    n = len(segment_ids)
    n_windows = n - lookback + 1
    if n_windows <= 0:
        return np.empty(0, dtype=np.int64)

    changed = np.zeros(n, dtype=np.int64)
    changed[1:] = (segment_ids[1:] != segment_ids[:-1]).astype(np.int64)
    cumulative = np.cumsum(changed)

    # Changes strictly inside window i are cumulative[i+lookback-1] - cumulative[i].
    inside = cumulative[lookback - 1:] - cumulative[:n_windows]
    no_gap = inside == 0

    target_ok = target_valid[lookback - 1:][:n_windows]
    return np.flatnonzero(no_gap & target_ok)


# ----------------------------------------------------------------------------
# Chronological split
# ----------------------------------------------------------------------------

@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    purge: np.ndarray
    n_total: int

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
            "purged": int(self.purge.size),
        }


def chronological_split(
    n_samples: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    purge: int,
) -> SplitIndices:
    """
    70/15/15 by default, in time order, with a `purge`-sample embargo between
    consecutive splits. Returns positional indices into the sample axis.
    """
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)

    train_end = n_train
    val_start = train_end + purge
    val_end = val_start + n_val
    test_start = val_end + purge

    if test_start >= n_samples:
        raise ValueError(
            f"Not enough samples ({n_samples}) for the requested split with "
            f"purge={purge}. Reduce the purge or fetch more data."
        )

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    test_idx = np.arange(test_start, n_samples, dtype=np.int64)
    purge_idx = np.concatenate([
        np.arange(train_end, val_start, dtype=np.int64),
        np.arange(val_end, test_start, dtype=np.int64),
    ])
    return SplitIndices(train_idx, val_idx, test_idx, purge_idx, n_samples)


# ----------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------

def build_scaler(kind: str = "standard"):
    from sklearn.preprocessing import RobustScaler, StandardScaler

    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler kind: {kind!r}")


def fit_scaler(X_train_2d: np.ndarray, kind: str = "standard"):
    """Fit on the TRAIN split only. Never call this with val/test data."""
    scaler = build_scaler(kind)
    scaler.fit(np.asarray(X_train_2d, dtype=np.float64))
    return scaler


def transform_scaled(X_2d: np.ndarray, scaler, clip_sigma: float | None = None) -> np.ndarray:
    """Apply the train-fitted scaler; returns float32. Optional symmetric clip."""
    out = scaler.transform(np.asarray(X_2d, dtype=np.float64))
    if clip_sigma is not None:
        np.clip(out, -clip_sigma, clip_sigma, out=out)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Lazy batching - the memory-efficient way to feed a Keras model
# ----------------------------------------------------------------------------

class SequenceBatcher:
    """
    Yields (X_batch, y_batch) 3D windows without materialising the whole tensor.

    Three separate things are kept separate on purpose:

        X_block        (n_block_rows, n_features)  the saved scaled 2D block
        y              (n_samples, n_horizons)     one target row PER SAMPLE
        start_offsets  (n_samples,)                where each window starts
                                                   inside X_block

    They cannot be collapsed into one index space. Gap-excluded windows leave
    HOLES in the sequence of window starts, so sample k's window is
    `X_block[start_offsets[k] : start_offsets[k] + lookback]` - never
    `X_block[k : k + lookback]`, which would splice across a hole and fabricate
    a window spanning missing market data.

    Only one batch is ever in memory, so a 300k x 64 x 49 tensor (several GB) is
    never allocated. Implements __len__ and __getitem__, so it can be passed
    straight to `keras.Model.fit`.

    `shuffle` permutes sample order WITHIN one split only. It never mixes splits
    and never reorders time inside a window.
    """

    def __init__(
        self,
        X_block: np.ndarray,
        y: np.ndarray,
        start_offsets: np.ndarray,
        *,
        lookback: int,
        batch_size: int = 512,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        self.X = np.asarray(X_block)
        self.y = np.asarray(y)
        self.starts = np.asarray(start_offsets, dtype=np.int64)
        self.lookback = int(lookback)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.n_samples = int(self.starts.size)

        if self.y.shape[0] != self.n_samples:
            raise ValueError(
                f"y has {self.y.shape[0]} rows but there are {self.n_samples} "
                f"window offsets - they must be aligned one-to-one."
            )
        if self.n_samples and self.starts.max() + self.lookback > len(self.X):
            raise ValueError(
                f"Window at offset {int(self.starts.max())} + lookback "
                f"{self.lookback} runs past the end of the block "
                f"({len(self.X)} rows)."
            )

    def __len__(self) -> int:
        return int(np.ceil(self.n_samples / self.batch_size))

    def order(self) -> np.ndarray:
        """Sample positions for this epoch (optionally shuffled)."""
        idx = np.arange(self.n_samples, dtype=np.int64)
        return self.rng.permutation(idx) if self.shuffle else idx

    def __getitem__(self, batch_index: int):
        order = self.order()
        chunk = order[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return self.materialise(chunk)

    def materialise(self, sample_index: np.ndarray):
        """Sample positions -> (k, LOOKBACK, n_features) float32 + (k, n_horizons) float32."""
        idx = np.asarray(sample_index, dtype=np.int64)
        starts = self.starts[idx]
        rows = starts[:, None] + np.arange(self.lookback, dtype=np.int64)[None, :]
        X = self.X[rows]
        y = self.y[idx]
        return (np.ascontiguousarray(X, dtype=np.float32),
                np.ascontiguousarray(y, dtype=np.float32))

    def as_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the WHOLE split as 3D tensors (only if it fits in RAM)."""
        return self.materialise(np.arange(self.n_samples, dtype=np.int64))
