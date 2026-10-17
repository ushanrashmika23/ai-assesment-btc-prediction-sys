"""
Cleaning, validation and continuity (gap) auditing of the raw Binance candles.

Design rules
------------
* UTC everywhere. The raw loader already normalises ms/us epochs to UTC; this
  module re-asserts it and refuses to continue if the dtype is not tz-aware UTC.
* Chronological ordering is enforced, never shuffled.
* Missing candles are NEVER fabricated. No reindex-and-interpolate, no
  forward-filling prices. A missing candle is reported and the series is cut
  into *segments* at every discontinuity.
* `segment_id` increments at every gap. Downstream code (sequences / targets)
  uses it to guarantee that a rolling window and its future target never
  straddle missing market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLEAN_CANDLES_CSV,
    INTERVAL_MINUTES,
    LARGE_GAP_MISSING_CANDLES,
    RAW_CANDLES_CSV,
    RAW_KEEP_COLUMNS,
)

STEP = pd.Timedelta(minutes=INTERVAL_MINUTES)


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------

@dataclass
class GapRecord:
    prev_timestamp: pd.Timestamp
    next_timestamp: pd.Timestamp
    missing_candles: int

    @property
    def is_large(self) -> bool:
        return self.missing_candles >= LARGE_GAP_MISSING_CANDLES


@dataclass
class ContinuityReport:
    n_rows: int = 0
    expected_rows: int = 0
    missing_candles: int = 0
    n_gaps: int = 0
    n_large_gaps: int = 0
    n_segments: int = 0
    largest_gap_missing: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    gaps: list[GapRecord] = field(default_factory=list)

    def render(self, max_listed: int = 15) -> str:
        pct = (100.0 * self.missing_candles / self.expected_rows) if self.expected_rows else 0.0
        lines = [
            "=" * 72,
            "CONTINUITY / GAP AUDIT",
            "=" * 72,
            f"candles present            : {self.n_rows:,}",
            f"candles expected on grid   : {self.expected_rows:,}",
            f"missing candles            : {self.missing_candles:,}  ({pct:.4f}%)",
            f"discontinuities (gaps)     : {self.n_gaps:,}",
            f"  of which large (>={LARGE_GAP_MISSING_CANDLES} candles) : {self.n_large_gaps:,}",
            f"largest gap (candles)      : {self.largest_gap_missing:,}",
            f"contiguous segments        : {self.n_segments:,}",
            f"first candle (UTC)         : {self.first_timestamp}",
            f"last candle  (UTC)         : {self.last_timestamp}",
        ]
        if self.gaps:
            lines.append("-" * 72)
            lines.append("gap detail (all discontinuities):")
            for g in self.gaps[:max_listed]:
                lines.append(
                    f"  {g.prev_timestamp} -> {g.next_timestamp}"
                    f"   missing={g.missing_candles}"
                    f"{'  [LARGE]' if g.is_large else ''}"
                )
            if len(self.gaps) > max_listed:
                lines.append(f"  ... and {len(self.gaps) - max_listed} more")
        lines.append("=" * 72)
        lines.append(
            "Policy: NO candle is fabricated. Every discontinuity starts a new\n"
            "        segment; sequences and targets may not cross a segment edge."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------

def load_raw(path=RAW_CANDLES_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw candles not found at {path}. Run src/data_download.py first."
        )
    return pd.read_csv(path)


def clean_candles(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Validate, type, sort, de-duplicate and segment the candle frame."""
    required = set(RAW_KEEP_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw candles are missing required columns: {sorted(missing)}")

    out = df.copy()

    # --- timestamps -> tz-aware UTC -----------------------------------------
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts and verbose:
        print(f"  [clean] dropped {n_bad_ts:,} rows with unparseable timestamps")
    out["timestamp"] = ts
    out = out.dropna(subset=["timestamp"])

    # --- numeric coercion ----------------------------------------------------
    numeric_cols = [c for c in RAW_KEEP_COLUMNS if c != "timestamp"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    if verbose and before - len(out):
        print(f"  [clean] dropped {before - len(out):,} rows with non-numeric OHLCV")

    # --- OHLC validity -------------------------------------------------------
    before = len(out)
    valid = (
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["high"] >= out["low"])
    )
    out = out.loc[valid]
    if verbose and before - len(out):
        print(f"  [clean] dropped {before - len(out):,} rows with invalid OHLC")

    for col in ("quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # --- chronological order + de-duplication --------------------------------
    out = out.sort_values("timestamp", kind="mergesort")
    before = len(out)
    out = out.drop_duplicates(subset="timestamp", keep="last")
    if verbose and before - len(out):
        print(f"  [clean] dropped {before - len(out):,} duplicate timestamps")

    out = out.reset_index(drop=True)

    # --- segment ids ---------------------------------------------------------
    out["segment_id"] = build_segment_ids(out["timestamp"])

    # Sanity: timestamps must be strictly increasing after de-duplication.
    if not out["timestamp"].is_monotonic_increasing:
        raise AssertionError("Timestamps are not monotonically increasing after cleaning")
    if out["timestamp"].duplicated().any():
        raise AssertionError("Duplicate timestamps survived cleaning")

    if verbose:
        print(f"  [clean] {len(out):,} clean candles, "
              f"{out['segment_id'].nunique():,} contiguous segments")
    return out


def build_segment_ids(timestamps: pd.Series) -> pd.Series:
    """0,0,0,1,1,... - increments at every discontinuity in the 15m grid."""
    deltas = timestamps.diff()
    new_segment = (deltas != STEP).fillna(False) & deltas.notna()
    return new_segment.cumsum().astype("int32")


# ----------------------------------------------------------------------------
# Gap detection
# ----------------------------------------------------------------------------

def detect_gaps(df: pd.DataFrame) -> ContinuityReport:
    ts = df["timestamp"]
    rep = ContinuityReport(
        n_rows=len(df),
        n_segments=int(df["segment_id"].nunique()) if "segment_id" in df else 1,
        first_timestamp=str(ts.iloc[0]) if len(ts) else None,
        last_timestamp=str(ts.iloc[-1]) if len(ts) else None,
    )
    if len(ts) < 2:
        return rep

    span = ts.iloc[-1] - ts.iloc[0]
    rep.expected_rows = int(round(span / STEP)) + 1
    rep.missing_candles = max(0, rep.expected_rows - len(df))

    deltas = ts.diff()
    bad = deltas.notna() & (deltas != STEP)
    for idx in np.flatnonzero(bad.to_numpy()):
        prev_ts = ts.iloc[idx - 1]
        next_ts = ts.iloc[idx]
        missing = int(round((next_ts - prev_ts) / STEP)) - 1
        rep.gaps.append(GapRecord(prev_ts, next_ts, max(0, missing)))

    rep.n_gaps = len(rep.gaps)
    rep.n_large_gaps = sum(1 for g in rep.gaps if g.is_large)
    rep.largest_gap_missing = max((g.missing_candles for g in rep.gaps), default=0)
    return rep


def segment_lengths(df: pd.DataFrame) -> pd.Series:
    return df.groupby("segment_id").size()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def preprocess(*, verbose: bool = True) -> tuple[pd.DataFrame, ContinuityReport]:
    """raw CSV -> validated, segmented clean CSV + continuity report."""
    print("Loading raw candles ...")
    raw = load_raw()

    print("Cleaning ...")
    clean = clean_candles(raw, verbose=verbose)

    report = detect_gaps(clean)
    print(report.render())

    CLEAN_CANDLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CANDLES_CSV, index=False)
    print(f"\nWrote {CLEAN_CANDLES_CSV}  ({len(clean):,} rows)")

    if report.n_gaps:
        print(
            "\nNOTE: discontinuities exist. They are handled by segment-aware\n"
            "      sequence generation and target invalidation - no prices are\n"
            "      interpolated or forward-filled."
        )
    return clean, report


if __name__ == "__main__":
    preprocess()
