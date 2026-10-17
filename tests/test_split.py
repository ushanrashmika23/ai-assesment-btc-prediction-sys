"""
T04 -- chronological ordering and purge boundaries.

The split is the one place a time-series model can leak without anyone
noticing, and it leaks in two distinct ways:

1. **Shuffling.** If any sample lands in an earlier split than a sample that
   precedes it in time, the model is trained on the future of its own test
   period. Checked here on the shipped ``timestamps_*.npy`` arrays.

2. **A too-thin purge.** A sample's LABEL is resolved from candles *after*
   its window, so two splits can be disjoint in sample index and still have
   the training labels resolved from candles the validation model sees as
   input. The purge has to exceed ``LOOKBACK - 1 + HORIZON_MAX`` rows, not
   just the horizon.

Both are asserted on the real artefacts, and ``chronological_split`` itself is
unit-tested so the boundary arithmetic is pinned independently of the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    HORIZON_MAX,
    LOOKBACK,
    PURGE_SAMPLES,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from sequences import chronological_split

CANDLE_MS = 15 * 60 * 1000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def split_timestamps():
    """The shipped per-sample target timestamps, or a skip if not built."""
    from _support import DATA_FINAL

    files = {name: DATA_FINAL / f"timestamps_{name}.npy"
             for name in ("train", "val", "test")}
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        pytest.skip(
            f"missing {missing} - run `python src/pipeline.py` to produce the "
            "split artefacts"
        )
    return {name: np.load(path) for name, path in files.items()}


def _row_index_by_timestamp(paths) -> dict[int, int]:
    """
    Absolute row index of every candle in the labelled frame, keyed by ms epoch.

    ``timestamps_all.npy`` holds the target timestamp of every row in order, so
    this maps a split sample back to the row its window actually ends on -- the
    only way to check the purge in ROWS rather than in samples.
    """
    ts_all = np.load(paths.data_final / "timestamps_all.npy")
    return {int(t): i for i, t in enumerate(ts_all)}


# ---------------------------------------------------------------------------
# T04 on the real shipped artefacts
# ---------------------------------------------------------------------------

def test_T04_splits_are_strictly_chronological(split_timestamps):
    """
    T04 -- train < validation < test, in wall-clock time, with no interleaving
    inside a split either.
    """
    ts = split_timestamps
    for name, arr in ts.items():
        assert arr.size > 0, f"{name} split is empty"
        assert np.all(np.diff(arr) > 0), (
            f"the {name} split is not strictly increasing in time - "
            f"{int((np.diff(arr) <= 0).sum())} non-advancing step(s), so it "
            "was shuffled or reordered"
        )

    train_max = int(ts["train"].max())
    val_min, val_max = int(ts["val"].min()), int(ts["val"].max())
    test_min = int(ts["test"].min())

    assert train_max < val_min, (
        "T04 FAILED: the train split runs past the start of validation.\n"
        f"  last train target     : {train_max} ({_iso(train_max)})\n"
        f"  first validation target: {val_min} ({_iso(val_min)})"
    )
    assert val_max < test_min, (
        "T04 FAILED: the validation split runs past the start of test.\n"
        f"  last validation target: {val_max} ({_iso(val_max)})\n"
        f"  first test target     : {test_min} ({_iso(test_min)})"
    )

    # No timestamp may appear in two splits.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = np.intersect1d(ts[a], ts[b])
        assert overlap.size == 0, (
            f"T04 FAILED: {overlap.size} timestamp(s) appear in BOTH the {a} "
            f"and {b} splits; first is {int(overlap[0])} ({_iso(int(overlap[0]))})"
        )

    print(
        f"T04 chronology OK: train {_iso(int(ts['train'].min()))} .. "
        f"{_iso(train_max)} | val {_iso(val_min)} .. {_iso(val_max)} | "
        f"test {_iso(test_min)} .. {_iso(int(ts['test'].max()))}"
    )


def test_T04_purge_separates_the_splits_in_rows(paths, split_timestamps):
    """
    T04 -- the purge boundary requirement, measured in ROWS on the real data.

    Three things must hold at each boundary:

    * the row gap is at least ``PURGE_SAMPLES`` candles (the samples that were
      discarded);
    * the LAST train label (its target row, plus HORIZON_MAX forward candles)
      resolves strictly before the FIRST validation window's opening row;
    * the train and validation windows therefore share no candle at all.
    """
    ts_all_path = paths.data_final / "timestamps_all.npy"
    if not ts_all_path.exists():
        pytest.skip("data/final/timestamps_all.npy not present")

    row_of = _row_index_by_timestamp(paths)
    ts = split_timestamps

    for earlier, later in (("train", "val"), ("val", "test")):
        last_target_row = row_of[int(ts[earlier].max())]
        first_window_end_row = row_of[int(ts[later].min())]
        first_window_start_row = first_window_end_row - (LOOKBACK - 1)

        row_gap = first_window_end_row - last_target_row
        label_reach = last_target_row + HORIZON_MAX

        assert row_gap >= PURGE_SAMPLES, (
            f"T04 FAILED: only {row_gap} rows separate {earlier} from {later}, "
            f"but PURGE_SAMPLES is {PURGE_SAMPLES}"
        )
        assert label_reach < first_window_start_row, (
            f"T04 FAILED: the last {earlier} label is resolved from a candle "
            f"inside the first {later} window.\n"
            f"  last {earlier} target row     : {last_target_row}\n"
            f"  its label reaches row        : {label_reach} "
            f"(+{HORIZON_MAX} forward candles)\n"
            f"  first {later} window starts at: {first_window_start_row}\n"
            f"  overlap                      : "
            f"{label_reach - first_window_start_row + 1} row(s)"
        )
        assert row_gap >= LOOKBACK, (
            f"T04 FAILED: {row_gap} rows of separation is less than one "
            f"lookback window ({LOOKBACK}), so a {earlier} window and a "
            f"{later} window would share raw candles"
        )

        clearance = first_window_start_row - label_reach - 1
        print(
            f"T04 purge {earlier}->{later}: {row_gap} rows "
            f"({row_gap * 15} min) >= purge {PURGE_SAMPLES}; last label "
            f"reaches row {label_reach}, first window starts row "
            f"{first_window_start_row} ({clearance} row(s) of clearance)"
        )


def test_T04_split_sizes_and_fractions_are_what_the_metadata_claims(
        paths, split_timestamps):
    """
    T04 -- the shipped split sizes must match the documented 70/15/15 and the
    metadata the API and the audit both read.
    """
    import json

    meta_path = paths.data_final / "dataset_metadata.json"
    if not meta_path.exists():
        pytest.skip("data/final/dataset_metadata.json not present")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sizes = meta["splits"]["sizes"]
    for name in ("train", "val", "test"):
        assert sizes[name] == split_timestamps[name].size, (
            f"dataset_metadata.json says the {name} split has {sizes[name]:,} "
            f"samples but timestamps_{name}.npy holds "
            f"{split_timestamps[name].size:,}"
        )

    total = sum(sizes[n] for n in ("train", "val", "test")) + sizes["purged"]
    for name, fraction in (("train", TRAIN_FRACTION), ("val", VAL_FRACTION),
                           ("test", TEST_FRACTION)):
        got = sizes[name] / total
        assert abs(got - fraction) < 0.01, (
            f"{name} is {got:.3%} of the {total:,} samples, not the declared "
            f"{fraction:.0%}"
        )

    assert sizes["purged"] == 2 * PURGE_SAMPLES, (
        f"{sizes['purged']} samples were purged but two boundaries of "
        f"PURGE_SAMPLES={PURGE_SAMPLES} is {2 * PURGE_SAMPLES}"
    )


# ---------------------------------------------------------------------------
# T04 on the split function itself
# ---------------------------------------------------------------------------

def test_T04_chronological_split_boundaries_are_exact():
    """
    Unit test for ``chronological_split``: the index arithmetic, the purge
    blocks and the sizes, pinned without reference to any dataset.
    """
    n, purge = 1000, 10
    split = chronological_split(n, train_fraction=0.70, val_fraction=0.15,
                                test_fraction=0.15, purge=purge)

    assert split.train[0] == 0 and split.train[-1] == 699, (
        f"train is {split.train[0]}..{split.train[-1]}, expected 0..699"
    )
    assert split.val[0] == 710 and split.val[-1] == 859, (
        f"val is {split.val[0]}..{split.val[-1]}, expected 710..859"
    )
    assert split.test[0] == 870 and split.test[-1] == 999, (
        f"test is {split.test[0]}..{split.test[-1]}, expected 870..999"
    )

    assert set(split.purge.tolist()) == set(range(700, 710)) | set(range(860, 870))

    # Disjoint and complete.
    everything = np.concatenate([split.train, split.val, split.test, split.purge])
    assert np.array_equal(np.sort(everything), np.arange(n)), (
        "the four index blocks do not partition 0..n-1 exactly once"
    )
    # 700 train + 10 purge + 150 val + 10 purge + 130 test = 1000 samples.
    assert split.sizes() == {"train": 700, "val": 150, "test": 130, "purged": 20}, (
        f"the split sizes are {split.sizes()}; expected "
        "{'train': 700, 'val': 150, 'test': 130, 'purged': 20} for n=1000 with "
        "purge=10 -- the blocks must partition 0..999 exactly once"
    )


def test_T04_chronological_split_rejects_a_purge_it_cannot_afford():
    """
    The guard the pipeline depends on: a purge too large for the sample count
    must raise rather than silently produce an empty or overlapping split.
    """
    with pytest.raises(ValueError, match="Not enough samples"):
        chronological_split(100, train_fraction=0.70, val_fraction=0.15,
                            test_fraction=0.15, purge=50)

    with pytest.raises(ValueError, match="sum to 1.0"):
        chronological_split(1000, train_fraction=0.70, val_fraction=0.15,
                            test_fraction=0.10, purge=10)


def test_T04_purge_translates_into_candles_of_separation():
    """
    Sweep the purge and pin the arithmetic that makes the guarantee true:
    purging P samples buys exactly P + 1 candles between the last ``earlier``
    sample and the first ``later`` one, and a label reaches
    ``LOOKBACK - 1 + HORIZON_MAX`` samples forward, so the property holds from
    that many purged samples onwards and not one sooner.
    """
    n = 5000
    required = LOOKBACK - 1 + HORIZON_MAX

    def separation_for(purge: int) -> int:
        split = chronological_split(n, train_fraction=0.70, val_fraction=0.15,
                                    test_fraction=0.15, purge=purge)
        return min(
            int(later[0]) - int(earlier[-1])
            for earlier, later in ((split.train, split.val),
                                   (split.val, split.test))
        )

    for purge in (0, 1, 10, 63, 68, 69, 70, 200):
        got = separation_for(purge)
        assert got == purge + 1, (
            f"purging {purge} samples produced {got} candles of separation; "
            f"expected exactly {purge + 1}"
        )

    assert separation_for(required - 1) <= required, (
        f"purging {required - 1} samples already satisfies the {required}-row "
        "requirement - the threshold below is not the real one"
    )
    assert separation_for(required) > required, (
        f"purging {required} samples leaves only {separation_for(required)} "
        f"rows of separation, which does not clear the {required}-row reach "
        "of the last training label"
    )

    # The value the project actually ships must clear it.
    assert PURGE_SAMPLES >= required, (
        f"PURGE_SAMPLES={PURGE_SAMPLES} is below the {required}-row minimum"
    )
    assert separation_for(PURGE_SAMPLES) > required, (
        "the configured purge does not separate the splits"
    )


def _iso(ms: int) -> str:
    import pandas as pd
    return str(pd.to_datetime(int(ms), unit="ms", utc=True))
