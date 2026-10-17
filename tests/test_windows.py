"""
T03 -- no generated window straddles a data gap.

A window that spans a discontinuity in the 15m grid would splice two
non-adjacent pieces of market history into one "16 hours of context" sample --
and, worse, the target of such a window would be resolved from a candle whose
relationship to the window's own close is fictional. The project forbids this
analytically (``sequences.valid_window_starts``) rather than by dropping rows
after the fact.

Two independent checks:

* **on the real shipped dataset** -- every window of all three splits, counted
  and verified against ``data/final/segment_ids.npy``. The shipped
  ``starts_<split>.npy`` hold offsets into each split's own block rather than
  absolute rows, so those offsets are first mapped back onto the labelled
  frame through ``timestamps_all.npy`` -- see ``_window_start_rows()``, which
  proves the mapping against both ``timestamps_<split>.npy`` and the shape of
  ``X_<split>.npy`` before anything is measured with it;
* **on the generator itself** -- a synthetic frame with an injected gap, where
  ``valid_window_starts`` is compared against a brute-force reference so the
  fast prefix-sum implementation cannot quietly disagree with the definition.

Neither check writes anything; the real dataset is opened read-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import LOOKBACK
from preprocessing import build_segment_ids
from sequences import valid_window_starts


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _brute_force_valid_starts(segment_ids: np.ndarray, lookback: int,
                              target_valid: np.ndarray) -> np.ndarray:
    """
    The definition, written the slow, obvious way: a window is valid iff every
    row it covers shares one segment and its target row has a usable target.

    O(n * lookback) -- fine for a few thousand rows, which is exactly why it
    can serve as the reference the vectorised version is checked against.
    """
    segment_ids = np.asarray(segment_ids)
    target_valid = np.asarray(target_valid, dtype=bool)
    n = len(segment_ids)
    out = []
    for i in range(0, n - lookback + 1):
        window = segment_ids[i:i + lookback]
        if window.min() != window.max():
            continue
        if not target_valid[i + lookback - 1]:
            continue
        out.append(i)
    return np.asarray(out, dtype=np.int64)


def _crossings(segment_ids: np.ndarray, starts: np.ndarray,
               lookback: int) -> np.ndarray:
    """
    Indices into ``starts`` whose window covers more than one segment.

    Segment ids are a non-decreasing step function, so "the whole window is one
    segment" is equivalent to "the segment id at the first row equals the one
    at the last row". The brute-force companion below double-checks that
    equivalence on a sample rather than trusting it.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)
    first = segment_ids[starts]
    last = segment_ids[starts + lookback - 1]
    return np.flatnonzero(first != last)


# ---------------------------------------------------------------------------
# T03 on the real shipped artefacts
# ---------------------------------------------------------------------------

def _window_start_rows(paths, split: str):
    """
    The absolute rows in ``segment_ids`` where each ``split`` window starts,
    plus the shipped per-block offsets they were derived from.

    ``starts_<split>.npy`` does not hold absolute rows. The pipeline saves each
    split as its own contiguous block -- ``X_<split>.npy`` -- and stores
    ``starts - starts[0]``: the offset of each window *within that block*,
    because gap-excluded windows leave holes in the index and those offsets are
    what make ``X[k : k + lookback]`` unambiguous. Feeding them to the global
    ``segment_ids`` is therefore wrong: it reads the row where the block
    happens to begin as if it were a segment edge. Train's block starts at row
    0, so it survives by luck; val starts at row 221,420 and test at 268,314,
    and both would report their very first window as a crossing.

    The mapping is recovered from the artefacts rather than assumed. A sample's
    target candle is ``timestamps_<split>[k]``, and ``timestamps_all.npy`` is
    the same clock over every row of the labelled frame, so the row a window
    ends on is the row its target timestamp sits at, and the window starts
    ``LOOKBACK - 1`` rows before that. Two independent facts then have to agree
    with the recovery, or this raises rather than measuring the wrong rows:

    * the recovered offsets equal the shipped ``starts_<split>.npy`` exactly;
    * the recovered span (first start .. last start + lookback) is exactly the
      row count of the shipped ``X_<split>.npy`` block.

    Returns ``(start_rows, segment_ids)``.
    """
    needed = {
        name: paths.data_final / name
        for name in ("segment_ids.npy", "timestamps_all.npy",
                     f"timestamps_{split}.npy", f"starts_{split}.npy",
                     f"X_{split}.npy")
    }
    missing = [name for name, p in needed.items() if not p.exists()]
    if missing:
        pytest.skip(
            f"missing {missing} - run `python src/pipeline.py` to produce the "
            "split artefacts"
        )

    segment_ids = np.load(needed["segment_ids.npy"])
    ts_all = np.load(needed["timestamps_all.npy"])
    ts_split = np.load(needed[f"timestamps_{split}.npy"])
    offsets = np.load(needed[f"starts_{split}.npy"])

    assert ts_all.size == segment_ids.size, (
        f"timestamps_all.npy has {ts_all.size:,} rows but segment_ids.npy has "
        f"{segment_ids.size:,}; they must index the same labelled frame"
    )
    assert np.all(np.diff(ts_all) > 0), (
        "timestamps_all.npy is not strictly increasing, so a target timestamp "
        "cannot be mapped back to a unique row"
    )

    target_rows = np.searchsorted(ts_all, ts_split)
    on_axis = ts_all[target_rows] == ts_split
    assert on_axis.all(), (
        f"{int((~on_axis).sum())} of the {ts_split.size:,} {split} target "
        "timestamps do not appear in timestamps_all.npy, so the split does not "
        "come from the frame segment_ids.npy describes"
    )

    start_rows = target_rows - (LOOKBACK - 1)
    recovered = start_rows - start_rows[0]
    assert np.array_equal(recovered, offsets), (
        f"the {split} block offsets recovered from the timestamps do not match "
        f"the shipped starts_{split}.npy ({int((recovered != offsets).sum())} "
        "of "
        f"{offsets.size:,} differ, first at sample "
        f"{int(np.argmax(recovered != offsets))}). Either the pipeline changed "
        "how it saves window starts or this recovery is wrong - do not measure "
        "anything with it until that is settled."
    )

    block_rows = int(start_rows[-1]) + LOOKBACK - int(start_rows[0])
    shipped_rows = int(np.load(needed[f"X_{split}.npy"], mmap_mode="r").shape[0])
    assert block_rows == shipped_rows, (
        f"the windows span {block_rows:,} rows of the labelled frame but "
        f"X_{split}.npy holds {shipped_rows:,} - the offsets are not into that "
        "block"
    )

    return start_rows, segment_ids


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_T03_no_window_crosses_a_gap_in_the_shipped_dataset(split, paths):
    """
    T03 -- check every generated window for gap crossing.

    Expected: 0 crossings. The number of windows checked is reported, and is
    asserted to be large, because "0 out of 0" would also be 0 crossings.
    """
    start_rows, segment_ids = _window_start_rows(paths, split)
    assert start_rows.size > 0, f"the {split} split holds no windows"

    n_rows = len(segment_ids)
    within_bounds = start_rows + LOOKBACK <= n_rows
    assert within_bounds.all(), (
        f"{int((~within_bounds).sum())} window(s) in the {split} split run past "
        f"the end of segment_ids ({n_rows:,} rows); largest start "
        f"{int(start_rows.max())} + lookback {LOOKBACK}"
    )

    bad = _crossings(segment_ids, start_rows, LOOKBACK)
    n_checked = int(start_rows.size)

    if bad.size:
        i = int(bad[0])
        s = int(start_rows[i])
        window = segment_ids[s:s + LOOKBACK]
        raise AssertionError(
            f"T03 FAILED: {bad.size} of {n_checked:,} windows in the {split} "
            f"split straddle a data gap (expected 0).\n"
            f"  first bad window: sample {i}, rows [{s}, {s + LOOKBACK}) of the "
            f"labelled frame\n"
            f"  segment ids in it: {window.tolist()}\n"
            f"  a window must sit inside one contiguous run of real candles"
        )

    # Independent spot check of the O(1) segment test against the O(n) one, so
    # a future change to how segment ids are packed cannot silently break the
    # equivalence the fast path relies on.
    rng = np.random.default_rng(0)
    sample = rng.choice(start_rows, size=min(500, start_rows.size), replace=False)
    for s in sample:
        s = int(s)
        window = segment_ids[s:s + LOOKBACK]
        assert window.min() == window.max(), (
            f"segment-id packing check disagrees with the fast path: window at "
            f"row {s} holds segments {sorted(set(window.tolist()))}"
        )

    print(
        f"T03[{split}] {n_checked:,} windows checked, 0 crossings "
        f"(lookback {LOOKBACK}, {n_rows:,} labelled rows, "
        f"{int(segment_ids.max()) + 1:,} contiguous segments, block starts at "
        f"row {int(start_rows[0]):,})"
    )


def test_T03_total_windows_checked_across_all_splits(paths):
    """
    T03 -- the headline count: how many windows were actually examined, and
    that the three splits together are what the pipeline claims to have
    produced.
    """
    total = 0
    per_split: dict[str, int] = {}
    for split in ("train", "val", "test"):
        start_rows, segment_ids = _window_start_rows(paths, split)
        bad = _crossings(segment_ids, start_rows, LOOKBACK)
        assert bad.size == 0, (
            f"{split}: {bad.size} window(s) straddle a segment edge"
        )
        per_split[split] = int(start_rows.size)
        total += int(start_rows.size)

    assert total > 100_000, (
        f"only {total:,} windows were checked across the three splits - the "
        "shipped dataset should hold a few hundred thousand, so this looks "
        "like a truncated or unbuilt artefact"
    )
    assert per_split["train"] > per_split["val"] > 0
    assert per_split["train"] > per_split["test"] > 0
    assert total == sum(per_split.values())

    # The three blocks are disjoint runs of the labelled frame, in order: that
    # is what makes "train, then val, then test" true of the shipped artefacts
    # and not merely of the index arithmetic.
    blocks = {}
    for split in ("train", "val", "test"):
        rows, _ids = _window_start_rows(paths, split)
        blocks[split] = (int(rows[0]), int(rows[-1]) + LOOKBACK)
    assert blocks["train"][1] <= blocks["val"][0], (
        f"the train block ends at row {blocks['train'][1]:,} but validation "
        f"starts at row {blocks['val'][0]:,} - the blocks overlap"
    )
    assert blocks["val"][1] <= blocks["test"][0], (
        f"the validation block ends at row {blocks['val'][1]:,} but test starts "
        f"at row {blocks['test'][0]:,} - the blocks overlap"
    )

    print(
        f"T03 TOTAL: {total:,} windows checked, 0 crossings  "
        f"(train {per_split['train']:,} | val {per_split['val']:,} | "
        f"test {per_split['test']:,})"
    )


# ---------------------------------------------------------------------------
# T03 on the generator itself
# ---------------------------------------------------------------------------

def test_T03_window_generator_matches_the_brute_force_definition():
    """
    Inject a gap into an otherwise regular grid and check that
    ``valid_window_starts`` returns exactly the windows a naive O(n*lookback)
    scan would accept -- no more, no fewer.

    This is the test that would catch a prefix-sum off-by-one; the real-dataset
    test above cannot, because it only ever sees the *output* of the same
    function.
    """
    n = 400
    lookback = 16
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))

    # Two gaps: a one-candle hole (small) and a five-candle hole (large).
    # The timestamps are shifted, exactly as the project's own audit does it,
    # so build_segment_ids() sees a genuine discontinuity in the grid.
    ts.iloc[120:] = ts.iloc[120:] + pd.Timedelta(minutes=15)     # 1 missing
    ts.iloc[250:] = ts.iloc[250:] + pd.Timedelta(minutes=75)     # 5 missing

    segment_ids = build_segment_ids(ts).to_numpy()
    assert segment_ids.max() == 2, (
        f"expected 3 segments after injecting 2 gaps, got "
        f"{segment_ids.max() + 1}"
    )

    target_valid = np.ones(n, dtype=bool)
    fast = valid_window_starts(segment_ids, target_valid, lookback)
    reference = _brute_force_valid_starts(segment_ids, lookback, target_valid)

    assert np.array_equal(fast, reference), (
        "valid_window_starts disagreed with the brute-force definition.\n"
        f"  fast path  : {fast.size} windows\n"
        f"  brute force: {reference.size} windows\n"
        f"  only in fast: {sorted(set(fast.tolist()) - set(reference.tolist()))[:10]}\n"
        f"  only in ref : {sorted(set(reference.tolist()) - set(fast.tolist()))[:10]}"
    )

    # The gaps must actually remove windows, otherwise this test would pass
    # even if valid_window_starts simply returned every start.
    n_candidate = n - lookback + 1
    assert fast.size < n_candidate, (
        f"injecting two gaps removed no windows ({fast.size} of "
        f"{n_candidate} candidates survive) - the fixture is not exercising "
        "the gap logic"
    )
    assert _crossings(segment_ids, fast, lookback).size == 0, (
        "the generator returned a window that crosses a segment edge"
    )
    # ...and every window it dropped really is a crossing one.
    dropped = np.setdiff1d(np.arange(n_candidate, dtype=np.int64), fast)
    assert dropped.size and _crossings(segment_ids, dropped, lookback).size == dropped.size, (
        "windows were excluded that do NOT cross a gap - the generator is "
        "dropping valid data"
    )

    print(
        f"T03[synthetic] {n_candidate} candidate windows, {fast.size} valid, "
        f"{dropped.size} excluded - all {dropped.size} excluded windows cross "
        f"a gap, 0 of the {fast.size} kept ones do"
    )


def test_T03_degenerate_inputs_are_rejected_or_empty():
    """
    The boundary behaviour the pipeline relies on: too few rows is an error,
    an empty result is an empty array (not a crash), and a target that is
    invalid everywhere yields no windows.
    """
    segment_ids = np.zeros(50, dtype=np.int64)

    with pytest.raises(ValueError):
        from sequences import make_sequences
        make_sequences(np.zeros((10, 3)), lookback=16)

    empty = valid_window_starts(segment_ids, np.ones(50, dtype=bool), lookback=64)
    assert empty.size == 0, "a lookback longer than the frame must yield no windows"

    none_valid = valid_window_starts(segment_ids, np.zeros(50, dtype=bool), lookback=16)
    assert none_valid.size == 0, (
        "windows were returned even though no target is usable"
    )
