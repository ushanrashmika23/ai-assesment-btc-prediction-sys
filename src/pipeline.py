"""
End-to-end pipeline: raw Binance candles -> Colab-ready training dataset.

    download -> clean/segment -> features -> targets -> warmup drop
             -> window validity -> chronological split + purge
             -> fit scaler on TRAIN rows only -> thresholds from TRAIN only
             -> materialise final dataset + arrays + metadata

Run:  python src/pipeline.py            (reuses cached raw data)
      python src/pipeline.py --refresh  (re-downloads first)

This module never imports TensorFlow or PyTorch and never trains anything.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from config import (
    CLIP_SIGMA,
    DIRECTION_COLUMNS,
    FEATURES_CSV,
    FINAL_DATASET_CSV,
    FINAL_DIR,
    FINAL_FEATURE_NAMES,
    FINAL_METADATA,
    FINAL_SCALER,
    FINAL_SEGMENT_IDS,
    FINAL_STARTS_TEST,
    FINAL_STARTS_TRAIN,
    FINAL_STARTS_VAL,
    FINAL_TARGET_SCALER,
    FINAL_TS_ALL,
    FINAL_TS_TEST,
    FINAL_TS_TRAIN,
    FINAL_TS_VAL,
    FINAL_X_TEST,
    FINAL_X_TRAIN,
    FINAL_X_VAL,
    FINAL_YDIR_TEST,
    FINAL_YDIR_TRAIN,
    FINAL_YDIR_VAL,
    FINAL_Y_TEST,
    FINAL_Y_TRAIN,
    FINAL_Y_VAL,
    FUTURE_RETURN_COLUMNS,
    HORIZON_LABELS,
    HORIZONS,
    LABELLED_CSV,
    LOOKBACK,
    PURGE_SAMPLES,
    RANDOM_SEED,
    RAW_CANDLES_CSV,
    SAVE_3D_ARRAYS,
    SCALER_KIND,
    SCALE_TARGETS,
    SEQUENCE_STRIDE,
    TEST_FRACTION,
    THRESHOLD_MODE,
    TRAIN_FRACTION,
    VAL_FRACTION,
    WARMUP_ROWS,
)
from config import (
    FINAL_YREGIME_TEST,
    FINAL_YREGIME_TRAIN,
    FINAL_YREGIME_VAL,
    REGIME_COLUMNS,
    REGIME_ENABLED,
    REGIME_HORIZONS,
)
from features import FEATURE_COLUMNS, FEATURE_GROUPS, build_features
from preprocessing import build_segment_ids, detect_gaps, preprocess
from sequences import (
    SequenceBatcher,
    chronological_split,
    fit_scaler,
    transform_scaled,
    valid_window_starts,
)
from targets import (
    add_direction_columns,
    add_future_returns,
    classify_returns,
    compute_thresholds,
    return_distribution_report,
    thresholds_to_jsonable,
)
from regime import (
    add_regime_label_columns,
    add_regime_ratios,
    classify_regime,
    compute_regime_thresholds,
    regime_thresholds_to_jsonable,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _jsonable(obj):
    """numpy / pandas -> plain python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if pd.isna(obj) if not isinstance(obj, (dict, list, str)) else False:
        return None
    return obj


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def run(*, refresh: bool = False, save_3d: bool = SAVE_3D_ARRAYS) -> dict:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. acquire ---------------------------------------------------------
    _banner("STEP 1/8 - DATA ACQUISITION")
    if refresh or not RAW_CANDLES_CSV.exists():
        from data_download import download_all
        download_all()
    else:
        print(f"Reusing cached raw data: {RAW_CANDLES_CSV}")
        print("(pass --refresh to re-download from Binance)")

    # -- 2. clean + segment -------------------------------------------------
    _banner("STEP 2/8 - CLEANING, VALIDATION AND GAP AUDIT")
    clean, continuity = preprocess()

    # -- 3. features --------------------------------------------------------
    _banner("STEP 3/8 - FEATURE ENGINEERING")
    featured = build_features(clean)

    # Regime ratios, computed HERE on the CLEAN CANDLE SERIES - before any row
    # is dropped, and against the segment ids that mark genuine market-data
    # gaps.
    #
    # WHY THIS FRAME AND NOT A LATER ONE. Two earlier placements were wrong:
    #
    #   * On the LABELLED frame but before the row drops (the original code) -
    #     the segment ids were stale, so the forward-window mask did not know
    #     about holes the drops had yet to make. Audit check 21 caught it.
    #   * On the post-drop frame, re-running until it settled - this does not
    #     terminate. Dropping a row whose forward window crosses a hole makes
    #     that hole one row wider, which invalidates the row behind it, which
    #     widens the hole again. Each pass ate ~350 more rows and would have
    #     consumed the dataset.
    #
    # On the clean series there is nothing to iterate: every candle is present,
    # the only discontinuities are real market gaps, and the answer cannot
    # depend on what later steps choose to drop. A row's label then describes
    # the actual market over the actual next h candles - which is what a label
    # should describe. It is also independently checkable: audit check 21
    # recomputes these columns straight from the clean candle CSV.
    if REGIME_ENABLED:
        _banner("STEP 2b/8 - MARKET-REGIME (ACTIVITY) TARGETS")
        regime_frame = add_regime_ratios(clean)

    # -- 4. targets ---------------------------------------------------------
    _banner("STEP 4/8 - TARGET ENGINEERING (future returns + activity regime)")
    labelled = add_future_returns(featured)

    if REGIME_ENABLED:
        # Features and targets are row-preserving transforms of `clean`, so the
        # regime columns line up positionally. Asserted rather than assumed.
        assert len(regime_frame) == len(labelled), "row count changed"
        assert np.array_equal(regime_frame["timestamp"].to_numpy(),
                              labelled["timestamp"].to_numpy()), "row order"
        for col in REGIME_COLUMNS + ["reference_absmove"]:
            labelled[col] = regime_frame[col].to_numpy()

    # -- 5. warmup + NaN handling ------------------------------------------
    _banner("STEP 5/8 - WARMUP AND MISSING-VALUE HANDLING")
    n_before = len(labelled)
    work = labelled.iloc[WARMUP_ROWS:].reset_index(drop=True)
    print(f"  dropped first {WARMUP_ROWS:,} warmup rows (indicator convergence)")

    feature_nan = work[FEATURE_COLUMNS].isna().sum()
    if feature_nan.sum():
        print(f"  {int(feature_nan.sum()):,} residual feature NaNs across "
              f"{int((feature_nan > 0).sum())} columns")
        for col, cnt in feature_nan[feature_nan > 0].items():
            print(f"      {col}: {cnt:,}")
    work = work.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"  rows after feature NaN drop : {len(work):,}")

    # The final `HORIZON_MAX` rows have no future close -> unusable as samples.
    target_ok = work[FUTURE_RETURN_COLUMNS].notna().all(axis=1).to_numpy()
    work = work.loc[target_ok].reset_index(drop=True)
    print(f"  rows after target NaN drop  : {len(work):,}")

    print(f"  total rows removed          : {n_before - len(work):,}")

    # Re-derive segment ids from the timestamps of the SURVIVING rows.
    #
    # This matters: dropping a row (feature NaN, or a gap-crossing target) makes
    # two non-adjacent candles neighbours in the frame. The inherited segment_id
    # would not notice, and a rolling window or a target could then silently span
    # the hole. Recomputing from `timestamp` turns EVERY discontinuity - Binance
    # gap or dropped row alike - into a hard segment boundary.
    n_segments_before = int(work["segment_id"].nunique())
    work["segment_id"] = build_segment_ids(work["timestamp"])
    n_segments_after = int(work["segment_id"].nunique())
    if n_segments_after != n_segments_before:
        print(f"  segment ids recomputed after row drops: "
              f"{n_segments_before:,} -> {n_segments_after:,} contiguous runs "
              f"(holes created by dropped rows are now boundaries too)")

    # The regime columns were already computed (step 2b) and merged (step 4);
    # here we only drop the rows that never had a full window.
    #
    # The segment-id recompute above is NOT used to re-derive them. It cannot
    # be: a label recomputed against these post-drop boundaries would be
    # measured over a frame with holes in it, and every drop would then widen a
    # hole and invalidate the row behind it - an iteration that does not
    # settle. The labels describe the real market over the real candles, which
    # is what they should describe, and audit check 21 verifies them directly
    # against the clean candle CSV rather than against this frame.
    if REGIME_ENABLED:
        # Single pass, no iteration. The ratios were computed on the clean
        # candle series, so this cannot cascade: dropping these rows cannot
        # invalidate a neighbour's label, because no label was computed from
        # the frame the drops produce.
        #
        # What is dropped: the first REF_WINDOW-1 rows (no trailing window yet)
        # and the last HORIZON_MAX rows (no forward window), plus any row whose
        # window would have crossed a real market gap. A row with a missing
        # ratio would otherwise be silently labelled QUIET.
        ok = work[REGIME_COLUMNS].notna().all(axis=1).to_numpy()
        n_bad = int((~ok).sum())
        if n_bad:
            print(f"  rows without a full regime window: {n_bad:,} "
                  f"(dropped - they would otherwise be labelled QUIET)")
        work = work.loc[ok].reset_index(drop=True)

    if len(work) < LOOKBACK + 1000:
        raise RuntimeError(
            f"Only {len(work):,} usable rows remain - too few to build "
            f"{LOOKBACK}-candle sequences. Check the download."
        )

    # -- 6. window validity + chronological split ---------------------------
    _banner("STEP 6/8 - SEQUENCE WINDOWS AND CHRONOLOGICAL SPLIT")

    X_2d_raw = work[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_2d = work[FUTURE_RETURN_COLUMNS].to_numpy(dtype=np.float64)
    regime_2d = (work[REGIME_COLUMNS].to_numpy(dtype=np.float64)
                 if REGIME_ENABLED else None)
    # Epoch milliseconds (UTC), stored as int64 so the .npy files stay plain
    # numeric arrays: no pickling, no timezone objects, portable everywhere.
    # Cast via datetime64[ms] rather than dividing an int64: pandas 3 may store
    # tz-aware datetimes at microsecond resolution, so the tick unit must not
    # be assumed.
    timestamps = (
        work["timestamp"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy()
        .astype("datetime64[ms]")
        .astype(np.int64)
    )
    segment_ids = work["segment_id"].to_numpy()

    # Every row in `work` already has a complete target set (filtered above).
    starts_all = valid_window_starts(
        segment_ids, np.ones(len(work), dtype=bool), LOOKBACK)
    if SEQUENCE_STRIDE > 1:
        starts_all = starts_all[::SEQUENCE_STRIDE]
    print(f"  candidate windows           : {len(work) - LOOKBACK + 1:,}")
    print(f"  valid windows (no gap, target present): {starts_all.size:,}")

    split = chronological_split(
        starts_all.size,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        purge=PURGE_SAMPLES,
    )
    sizes = split.sizes()
    print(f"  purge/embargo between splits: {PURGE_SAMPLES} samples "
          f"(= lookback {LOOKBACK} + max horizon {max(HORIZONS)}, i.e. no train "
          f"label or window touches a validation candle)")
    print(f"  train {sizes['train']:,} | val {sizes['val']:,} | "
          f"test {sizes['test']:,} | purged {sizes['purged']:,}")

    train_starts = starts_all[split.train]
    val_starts = starts_all[split.val]
    test_starts = starts_all[split.test]

    def target_row(idx: np.ndarray) -> np.ndarray:
        """Each sample's target is the target of its LAST candle."""
        return idx + LOOKBACK - 1

    ts_train = timestamps[target_row(train_starts)]
    ts_val = timestamps[target_row(val_starts)]
    ts_test = timestamps[target_row(test_starts)]

    def fmt(ms: int) -> str:
        return str(pd.to_datetime(int(ms), unit="ms", utc=True))

    # -- 7. scaling (TRAIN only) + thresholds (TRAIN only) ------------------
    _banner("STEP 7/8 - SCALING AND THRESHOLDS (fitted on TRAIN only)")

    # Train windows occupy rows [train_starts[0], train_starts[-1] + LOOKBACK).
    train_block_end = int(train_starts[-1]) + LOOKBACK
    first_val_start = int(val_starts[0])

    # The last TRAIN label is resolved from close[t] and close[t + h]; the
    # furthest candle it touches is LOOKBACK-1 + HORIZON_MAX rows past the last
    # train window's start. That reach must stop before the first validation
    # FEATURE row, or training labels would be computed from candles the
    # validation inputs also summarise.
    label_reach = int(train_starts[-1]) + LOOKBACK - 1 + max(HORIZONS)
    if label_reach >= first_val_start or train_block_end > first_val_start:
        raise AssertionError(
            f"Split separation is too small: last train label reaches row "
            f"{label_reach:,}, last train window ends at row {train_block_end:,}, "
            f"but validation starts at row {first_val_start:,}. "
            f"Increase PURGE_SAMPLES (currently {PURGE_SAMPLES})."
        )

    # The scaler is fitted on the rows that make up the TRAINING windows, and on
    # nothing else - see the assertion above.
    scaler = fit_scaler(X_2d_raw[:train_block_end], kind=SCALER_KIND)
    print(f"  scaler fitted on rows [0, {train_block_end:,}) - every row of "
          f"every TRAIN window and nothing beyond")
    print(f"  last train label touches   : row {label_reach:,} "
          f"(target candle + {max(HORIZONS)} forward candles)")
    clearance = first_val_start - label_reach - 1
    print(f"  first validation feature   : row {first_val_start:,} "
          f"({clearance:,} row{'s' if clearance != 1 else ''} of clearance, "
          f"purge={PURGE_SAMPLES} samples)")
    print(f"  scaler kind                 : {SCALER_KIND}")

    def scaled_block(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Scaled contiguous 2D block covering these windows, plus each sample's
        window-start offset WITHIN that block.

        Saving the block (not one row per sample) matters: gap-excluded windows
        leave HOLES in the start index, so slicing `X[k : k+64]` would silently
        splice across a hole and fabricate a window spanning a data gap. The
        offsets make that impossible - always index via the offsets.
        """
        lo, hi = int(starts[0]), int(starts[-1]) + LOOKBACK
        block = transform_scaled(X_2d_raw[lo:hi], scaler, CLIP_SIGMA)
        return block, (starts - lo).astype(np.int64)

    X_train, s_train = scaled_block(train_starts)
    X_val, s_val = scaled_block(val_starts)
    X_test, s_test = scaled_block(test_starts)
    print(f"  scaled blocks               : X {X_train.shape} {X_val.shape} "
          f"{X_test.shape}, clip=+/-{CLIP_SIGMA} sigma")
    print(f"  window offsets              : {s_train.size:,} / {s_val.size:,} / "
          f"{s_test.size:,} samples")

    y_train_raw = y_2d[target_row(train_starts)]
    y_val_raw = y_2d[target_row(val_starts)]
    y_test_raw = y_2d[target_row(test_starts)]

    # Directional thresholds come from the TRAIN target distribution only.
    thresholds = compute_thresholds(y_train_raw, mode=THRESHOLD_MODE)
    print(f"  threshold mode              : {THRESHOLD_MODE}")
    for h in HORIZONS:
        lo, hi = thresholds[h]
        print(f"      {HORIZON_LABELS[h]:>4}: [{lo:+.4f}%, {hi:+.4f}%]  "
              f"= [{lo * 100:+.2f}, {hi * 100:+.2f}] bps")

    ydir_train = classify_returns(y_train_raw, thresholds)
    ydir_val = classify_returns(y_val_raw, thresholds)
    ydir_test = classify_returns(y_test_raw, thresholds)

    # Regime thresholds: the TRAIN median of the forward/reference activity
    # ratio, per horizon. Median means the two classes are 50/50 on train by
    # construction, and - unlike the direction thresholds above - the balance
    # then HOLDS on val and test, because the ratio is self-normalising.
    regime_thresholds = None
    yreg_train = yreg_val = yreg_test = None
    if REGIME_ENABLED:
        r_train = regime_2d[target_row(train_starts)]
        r_val = regime_2d[target_row(val_starts)]
        r_test = regime_2d[target_row(test_starts)]
        regime_thresholds = compute_regime_thresholds(r_train)
        print(f"  regime threshold mode       : median of TRAIN ratio "
              f"(quantile {config.REGIME_THRESHOLD_QUANTILE})")
        for i, h in enumerate(REGIME_HORIZONS):
            print(f"      {HORIZON_LABELS[h]:>4}: ratio > "
                  f"{regime_thresholds[i]:.4f} => ACTIVE")
        yreg_train = classify_regime(r_train, regime_thresholds)
        yreg_val = classify_regime(r_val, regime_thresholds)
        yreg_test = classify_regime(r_test, regime_thresholds)

    # Optional target scaler - train statistics only.
    target_scaler = None
    if SCALE_TARGETS:
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler().fit(y_train_raw)

    # -- 8. save ------------------------------------------------------------
    _banner("STEP 8/8 - WRITING FINAL DELIVERABLES")

    # Full human-readable dataset (unscaled), with direction labels applied
    # using the frozen train thresholds.
    final_frame = add_direction_columns(work, thresholds)
    regime_label_columns = []
    if REGIME_ENABLED:
        final_frame = add_regime_label_columns(final_frame, regime_thresholds)
        regime_label_columns = [f"regime_{HORIZON_LABELS[h]}"
                                for h in REGIME_HORIZONS]
    ordered = (
        ["timestamp", "segment_id"]
        + FEATURE_COLUMNS
        + FUTURE_RETURN_COLUMNS
        + DIRECTION_COLUMNS
        + REGIME_COLUMNS
        + regime_label_columns
        + (["reference_absmove"] if REGIME_ENABLED else [])
    )
    final_frame = final_frame[ordered]
    float_cols = FEATURE_COLUMNS + FUTURE_RETURN_COLUMNS + REGIME_COLUMNS
    final_frame[float_cols] = final_frame[float_cols].round(8)
    final_frame.to_csv(FINAL_DATASET_CSV, index=False)
    print(f"  {FINAL_DATASET_CSV.name:<32} {len(final_frame):,} rows x "
          f"{final_frame.shape[1]} cols")

    # Intermediate artefacts (useful for inspection / notebooks 02-03)
    feature_only = work[["timestamp", "segment_id"] + FEATURE_COLUMNS]
    feature_only.round(8).to_csv(FEATURES_CSV, index=False)
    final_frame.round(8).to_csv(LABELLED_CSV, index=False)

    np.save(FINAL_X_TRAIN, X_train)
    np.save(FINAL_X_VAL, X_val)
    np.save(FINAL_X_TEST, X_test)
    np.save(FINAL_STARTS_TRAIN, s_train)
    np.save(FINAL_STARTS_VAL, s_val)
    np.save(FINAL_STARTS_TEST, s_test)
    np.save(FINAL_Y_TRAIN, y_train_raw.astype(np.float32))
    np.save(FINAL_Y_VAL, y_val_raw.astype(np.float32))
    np.save(FINAL_Y_TEST, y_test_raw.astype(np.float32))
    np.save(FINAL_YDIR_TRAIN, ydir_train)
    np.save(FINAL_YDIR_VAL, ydir_val)
    np.save(FINAL_YDIR_TEST, ydir_test)
    if REGIME_ENABLED:
        np.save(FINAL_YREGIME_TRAIN, yreg_train)
        np.save(FINAL_YREGIME_VAL, yreg_val)
        np.save(FINAL_YREGIME_TEST, yreg_test)
    np.save(FINAL_TS_TRAIN, ts_train)
    np.save(FINAL_TS_VAL, ts_val)
    np.save(FINAL_TS_TEST, ts_test)
    # Row-index reference for the leakage audit: the timestamp of every row of
    # the labelled frame, plus its segment id (so a window can be proven to lie
    # inside one contiguous run of real candles).
    np.save(FINAL_TS_ALL, timestamps)
    np.save(FINAL_SEGMENT_IDS, segment_ids.astype(np.int32))
    print("  X_{train,val,test}.npy        scaled 2D blocks (float32)")
    print("  starts_{train,val,test}.npy  window offsets into each block")
    print("  y_*.npy, y_dir_*.npy, y_regime_*.npy, timestamps_*.npy")
    print("  timestamps_all.npy, segment_ids.npy  (audit row index)")

    import pickle
    with open(FINAL_SCALER, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  {FINAL_SCALER.name:<32} ({type(scaler).__name__})")
    if target_scaler is not None:
        with open(FINAL_TARGET_SCALER, "wb") as fh:
            pickle.dump(target_scaler, fh)
        print(f"  {FINAL_TARGET_SCALER.name:<32} (StandardScaler)")

    with open(FINAL_FEATURE_NAMES, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "feature_groups": FEATURE_GROUPS,
                "target_columns": FUTURE_RETURN_COLUMNS,
                "direction_columns": DIRECTION_COLUMNS,
            },
            fh,
            indent=2,
        )

    # Optional 3D materialisation (off by default - several GB).
    if save_3d:
        batcher = SequenceBatcher(X_train, y_train_raw, s_train, lookback=LOOKBACK)
        np.save(FINAL_DIR / "X_train_3d.npy", batcher.as_dataset()[0])
        print("  X_train_3d.npy (materialised 3D tensor)")

    # -- metadata -----------------------------------------------------------
    regime_dist = None
    if REGIME_ENABLED:
        regime_dist = {}
        for name, yreg in (("train", yreg_train), ("val", yreg_val),
                           ("test", yreg_test)):
            regime_dist[name] = {}
            for i, h in enumerate(REGIME_HORIZONS):
                col = yreg[:, i]
                n = int((col >= 0).sum())
                regime_dist[name][HORIZON_LABELS[h]] = {
                    "n": n,
                    "QUIET": int((col == 0).sum()),
                    "ACTIVE": int((col == 1).sum()),
                    "QUIET_%": round(100.0 * float((col == 0).sum()) / n, 3) if n else None,
                    "ACTIVE_%": round(100.0 * float((col == 1).sum()) / n, 3) if n else None,
                }

    class_dist = {}
    for name, ydir in (("train", ydir_train), ("val", ydir_val), ("test", ydir_test)):
        class_dist[name] = {}
        for i, h in enumerate(HORIZONS):
            col = ydir[:, i]
            n = int((col >= 0).sum())
            class_dist[name][HORIZON_LABELS[h]] = {
                "n": n,
                "DOWN": int((col == 0).sum()),
                "NEUTRAL": int((col == 1).sum()),
                "UP": int((col == 2).sum()),
                "DOWN_%": round(100.0 * float((col == 0).sum()) / n, 3) if n else None,
                "NEUTRAL_%": round(100.0 * float((col == 1).sum()) / n, 3) if n else None,
                "UP_%": round(100.0 * float((col == 2).sum()) / n, 3) if n else None,
            }

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": config.SYMBOL,
        "interval": config.INTERVAL,
        "dataset": {
            "rows": int(len(final_frame)),
            "columns": int(final_frame.shape[1]),
            "n_features": len(FEATURE_COLUMNS),
            "n_targets": len(FUTURE_RETURN_COLUMNS),
            "n_direction_targets": len(DIRECTION_COLUMNS),
            "first_timestamp_utc": str(final_frame["timestamp"].iloc[0]),
            "last_timestamp_utc": str(final_frame["timestamp"].iloc[-1]),
            "csv": FINAL_DATASET_CSV.name,
            "float_rounding_decimals": 8,
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "target_columns": FUTURE_RETURN_COLUMNS,
        "direction_columns": DIRECTION_COLUMNS,
        "direction_classes": config.DIRECTION_CLASSES,
        "lookback": LOOKBACK,
        "lookback_hours": LOOKBACK * config.INTERVAL_MINUTES / 60.0,
        "sequence_stride": SEQUENCE_STRIDE,
        "horizons": {
            HORIZON_LABELS[h]: {"candles": h, "minutes": h * config.INTERVAL_MINUTES}
            for h in HORIZONS
        },
        "warmup_rows_dropped": WARMUP_ROWS,
        "splits": {
            "fractions": {
                "train": TRAIN_FRACTION,
                "val": VAL_FRACTION,
                "test": TEST_FRACTION,
            },
            "purge_samples": PURGE_SAMPLES,
            "sizes": sizes,
            "train": {
                "n_samples": sizes["train"],
                "first_target_ts_utc": fmt(ts_train[0]),
                "last_target_ts_utc": fmt(ts_train[-1]),
            },
            "val": {
                "n_samples": sizes["val"],
                "first_target_ts_utc": fmt(ts_val[0]),
                "last_target_ts_utc": fmt(ts_val[-1]),
            },
            "test": {
                "n_samples": sizes["test"],
                "first_target_ts_utc": fmt(ts_test[0]),
                "last_target_ts_utc": fmt(ts_test[-1]),
            },
        },
        "array_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train_raw.shape),
            "starts_train": list(s_train.shape),
            "note": ("X_*.npy are contiguous 2D blocks. Sample k's window is "
                     "X[start_k : start_k + lookback], where start_k comes from "
                     "starts_*.npy. Do NOT slice X[k : k+lookback] directly - "
                     "gap-excluded windows leave holes in the start index."),
        },
        "sequence_tensor_shape": {
            "note": "materialise per batch; see src/sequences.py::SequenceBatcher",
            "X_train_3d": [sizes["train"], LOOKBACK, len(FEATURE_COLUMNS)],
            "y_train": [sizes["train"], len(HORIZONS)],
        },
        "scaling": {
            "kind": SCALER_KIND,
            # NOTE: these three fields must describe the range ACTUALLY passed
            # to fit_scaler() below - rows [0, train_block_end) - and not the
            # first validation row. Recording first_val_start here would claim
            # the scaler saw the purge-gap rows, which it never does: the audit
            # cross-checks this range against the shipped blocks.
            "fitted_on": "2D rows making up the TRAIN windows, and nothing else",
            "fitted_on_row_range": [0, int(train_block_end)],
            "fitted_on_rows": int(train_block_end),
            "fit_range_ends_before_first_val_row": int(first_val_start),
            "clip_sigma": CLIP_SIGMA,
            "scaler_file": FINAL_SCALER.name,
            "target_scaler_file": FINAL_TARGET_SCALER.name if target_scaler else None,
            "target_scaler_mean": _jsonable(target_scaler.mean_) if target_scaler else None,
            "target_scaler_scale": _jsonable(target_scaler.scale_) if target_scaler else None,
            "targets_are_percent_not_scaled_on_disk": True,
        },
        "thresholds": {
            "mode": THRESHOLD_MODE,
            "neutral_quantile": config.NEUTRAL_QUANTILE,
            "derived_from": "TRAIN split only",
            "per_horizon": thresholds_to_jsonable(thresholds),
        },
        "regime_targets": {
            "enabled": bool(REGIME_ENABLED),
            "question": ("will the mean |15m return| over the next h candles "
                         "exceed its own trailing 24h average?"),
            "horizons": [HORIZON_LABELS[h] for h in REGIME_HORIZONS],
            "horizon_candles": list(REGIME_HORIZONS),
            "reference_window_candles": config.REGIME_REF_WINDOW,
            "ratio_columns": REGIME_COLUMNS,
            "classes": config.REGIME_CLASSES,
            "thresholds": (regime_thresholds_to_jsonable(regime_thresholds)
                           if REGIME_ENABLED else None),
            "derived_from": "TRAIN split only (median of the train ratio)",
            "why": ("self-normalising, so the class balance does NOT drift "
                    "between splits the way the absolute-percent direction "
                    "thresholds do"),
        } if REGIME_ENABLED else {"enabled": False},
        "class_distribution": class_dist,
        "regime_class_distribution": regime_dist,
        "continuity": {
            "candles_present": continuity.n_rows,
            "candles_expected": continuity.expected_rows,
            "missing_candles": continuity.missing_candles,
            "n_gaps": continuity.n_gaps,
            "n_large_gaps": continuity.n_large_gaps,
            "largest_gap_missing_candles": continuity.largest_gap_missing,
            "n_segments": continuity.n_segments,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "random_seed": RANDOM_SEED,
            "tensorflow_used_locally": False,
        },
    }

    with open(FINAL_METADATA, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(metadata), fh, indent=2)
    print(f"  {FINAL_METADATA.name:<32} (all reported statistics)")

    # -- console summary ----------------------------------------------------
    _banner("FINAL DATASET SUMMARY")
    print(f"rows               : {len(final_frame):,}")
    print(f"features           : {len(FEATURE_COLUMNS)}")
    print(f"return targets     : {len(FUTURE_RETURN_COLUMNS)}")
    print(f"direction targets  : {len(DIRECTION_COLUMNS)}")
    print(f"lookback           : {LOOKBACK} candles "
          f"({LOOKBACK * config.INTERVAL_MINUTES / 60:.0f}h)")
    print(f"horizons           : {', '.join(HORIZON_LABELS[h] for h in HORIZONS)}")
    print(f"train / val / test : {sizes['train']:,} / {sizes['val']:,} / {sizes['test']:,}"
          f"  (+{sizes['purged']:,} purged)")
    print(f"train period       : {fmt(ts_train[0])}  ->  {fmt(ts_train[-1])}")
    print(f"val period         : {fmt(ts_val[0])}  ->  {fmt(ts_val[-1])}")
    print(f"test period        : {fmt(ts_test[0])}  ->  {fmt(ts_test[-1])}")

    print("\nreturn distribution (full sample, percent):")
    dist = return_distribution_report(work)
    print(dist[["count", "mean_%", "std_%", "p05_%", "p50_%", "p95_%", "pct_positive"]]
          .to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\nclass balance per split (train thresholds applied):")
    for name in ("train", "val", "test"):
        print(f"  {name}:")
        for i, h in enumerate(HORIZONS):
            d = class_dist[name][HORIZON_LABELS[h]]
            print(f"      {HORIZON_LABELS[h]:>4}  DOWN {d['DOWN_%']:>6.2f}%   "
                  f"NEUTRAL {d['NEUTRAL_%']:>6.2f}%   UP {d['UP_%']:>6.2f}%")

    if REGIME_ENABLED:
        print("\nregime balance per split (train median threshold applied) - "
              "note how little it moves:")
        for name in ("train", "val", "test"):
            print(f"  {name}:")
            for h in REGIME_HORIZONS:
                d = regime_dist[name][HORIZON_LABELS[h]]
                print(f"      {HORIZON_LABELS[h]:>4}  QUIET {d['QUIET_%']:>6.2f}%   "
                      f"ACTIVE {d['ACTIVE_%']:>6.2f}%")
        print("  (the direction balance above slides from 30/40/30 on train to")
        print("   ~24/54/22 on test; the regime balance does not, because the")
        print("   ratio is normalised by the sample's own trailing activity)")

    print(f"\nAll deliverables in: {FINAL_DIR}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT 15m -> Colab training dataset")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download raw Binance data before processing")
    parser.add_argument("--save-3d", action="store_true",
                        help="also write materialised 3D tensors (large)")
    args = parser.parse_args()
    run(refresh=args.refresh, save_3d=args.save_3d)


if __name__ == "__main__":
    main()
