"""
Binance BTCUSDT 15m historical data acquisition.

Strategy (most efficient first):

  1. Bulk monthly ZIP archives from data.binance.vision - one HTTP request per
     month instead of one per 1000 candles.  ~110 requests covers 2017-08 to
     the present, versus ~320 REST round-trips.
  2. Daily ZIP archives for the current (still open) month.
  3. Binance REST /api/v3/klines for the last few candles that are not yet
     mirrored into the archive - paginated, rate-limit aware, retried.

Everything lands in one chronologically sorted, de-duplicated CSV.

No API credentials are required: this is public market data.

Robustness handled here
-----------------------
* timestamp unit switch  : Binance used milliseconds until 2024-12 and
                           MICROSECONDS from 2025-01 onwards.
* header / no-header     : some archives ship a header row, most do not.
* duplicate candles      : monthly / daily / REST ranges overlap by design.
* chronological ordering : final sort on the UTC timestamp.
* malformed rows         : non-numeric / short rows are dropped and counted.
* invalid OHLC           : non-positive prices and high<low are dropped.
* HTTP failures          : retried with exponential backoff, then skipped and
                           reported rather than silently producing a hole.

Gaps are NOT filled here.  Fabricating candles would corrupt the target
labels; gap detection and safe handling live in preprocessing.py.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from config import (
    BINANCE_KLINE_COLUMNS,
    FIRST_DATA_MONTH,
    HTTP_TIMEOUT,
    INTERVAL,
    INTERVAL_MINUTES,
    RAW_CANDLES_CSV,
    RAW_KEEP_COLUMNS,
    RAW_ZIP_DIR,
    REST_BACKOFF_SECONDS,
    REST_BASE,
    REST_LIMIT,
    REST_MAX_RETRIES,
    SYMBOL,
    VISION_BASE,
)

MS_THRESHOLD = 10**14   # anything >= this is microseconds, not milliseconds
SEC_THRESHOLD = 10**11  # anything <  this is seconds, not milliseconds


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

@dataclass
class DownloadReport:
    months_requested: int = 0
    months_ok: int = 0
    months_missing: list[str] = field(default_factory=list)
    days_requested: int = 0
    days_ok: int = 0
    days_missing: list[str] = field(default_factory=list)
    rest_rows: int = 0
    raw_rows: int = 0
    malformed_rows_dropped: int = 0
    invalid_ohlc_dropped: int = 0
    duplicate_rows_dropped: int = 0
    final_rows: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    def render(self) -> str:
        lines = [
            "=" * 72,
            "BINANCE DOWNLOAD REPORT",
            "=" * 72,
            f"symbol / interval          : {SYMBOL} / {INTERVAL}",
            f"monthly archives           : {self.months_ok}/{self.months_requested} ok",
            f"daily archives (open month): {self.days_ok}/{self.days_requested} ok",
            f"rows from REST top-up      : {self.rest_rows:,}",
            f"raw rows read              : {self.raw_rows:,}",
            f"malformed rows dropped     : {self.malformed_rows_dropped:,}",
            f"invalid OHLC rows dropped  : {self.invalid_ohlc_dropped:,}",
            f"duplicate rows dropped     : {self.duplicate_rows_dropped:,}",
            f"final rows written         : {self.final_rows:,}",
            f"first candle (UTC)         : {self.first_timestamp}",
            f"last candle  (UTC)         : {self.last_timestamp}",
        ]
        if self.months_missing:
            lines.append(f"missing months             : {', '.join(self.months_missing)}")
        if self.days_missing:
            lines.append(f"missing days               : {', '.join(self.days_missing)}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------

def _http_get(url: str, *, params: dict | None = None, as_json: bool = False):
    """GET with retries + exponential backoff. Returns None if unreachable."""
    last_err: Exception | None = None
    for attempt in range(REST_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 404:
                return None                      # genuinely absent (e.g. future month)
            if resp.status_code == 429:          # rate limited
                time.sleep(REST_BACKOFF_SECONDS * (2**attempt))
                continue
            resp.raise_for_status()
            return resp.json() if as_json else resp.content
        except Exception as exc:                 # noqa: BLE001 - report and retry
            last_err = exc
            time.sleep(REST_BACKOFF_SECONDS * (2**attempt))
    print(f"    [warn] giving up on {url}: {last_err}")
    return None


# ----------------------------------------------------------------------------
# Timestamp normalisation
# ----------------------------------------------------------------------------

def normalise_epoch_to_ms(values: pd.Series) -> pd.Series:
    """
    Binance has used seconds, milliseconds and (since 2025-01) microseconds for
    kline open times.  Collapse all three onto milliseconds.
    """
    ts = pd.to_numeric(values, errors="coerce").astype("float64")
    ms = ts.copy()
    ms = ms.where(ts < MS_THRESHOLD, ts / 1000.0)      # microseconds -> ms
    ms = ms.where(ms >= SEC_THRESHOLD, ms * 1000.0)    # seconds      -> ms
    return ms.round().astype("Int64")


def to_utc_datetime(ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ms, unit="ms", utc=True)


# ----------------------------------------------------------------------------
# Archive parsing
# ----------------------------------------------------------------------------

def _parse_kline_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Take a raw headerless frame and return typed, UTC, validated candles."""
    n_in = len(df)
    if df.empty:
        return df

    # Some archives ship a header row; drop it if the first cell is not numeric.
    first_cell = str(df.iloc[0, 0])
    if not first_cell.replace(".", "", 1).replace("-", "", 1).isdigit():
        df = df.iloc[1:]

    if df.shape[1] < 12:
        # Pad short rows rather than trusting positional alignment.
        for i in range(df.shape[1], 12):
            df[i] = np.nan
    df = df.iloc[:, :12]
    df.columns = BINANCE_KLINE_COLUMNS

    numeric_cols = [c for c in BINANCE_KLINE_COLUMNS if c != "ignore"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    malformed = before - len(df)

    df["timestamp"] = to_utc_datetime(normalise_epoch_to_ms(df["open_time"]))
    df = df.dropna(subset=["timestamp"])

    # OHLC sanity: strictly positive prices, high >= low, high/low bracket open/close.
    ok = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
        & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
        & (df["volume"] >= 0)
    )
    n_bad = int((~ok).sum())
    df = df.loc[ok]

    df["number_of_trades"] = df["number_of_trades"].fillna(0)
    for col in ("quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"):
        df[col] = df[col].fillna(0.0)

    out = df[RAW_KEEP_COLUMNS].copy()
    out["_source"] = source
    out.attrs["malformed"] = malformed
    out.attrs["invalid_ohlc"] = n_bad
    out.attrs["rows_in"] = n_in
    return out


def _read_zip_csv(payload: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(payload))
    name = zf.namelist()[0]
    text = zf.read(name).decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), header=None, dtype=str, on_bad_lines="skip")


# ----------------------------------------------------------------------------
# Archive URL builders
# ----------------------------------------------------------------------------

def monthly_url(year: int, month: int) -> str:
    return f"{VISION_BASE}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{year:04d}-{month:02d}.zip"


def daily_url(day: date) -> str:
    return f"{VISION_BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{day:%Y-%m-%d}.zip"


# ----------------------------------------------------------------------------
# Downloaders
# ----------------------------------------------------------------------------

def month_range(first: str, last_exclusive: date) -> list[tuple[int, int]]:
    """['2017-08' .. ) -> [(2017, 8), (2017, 9), ...] up to last_exclusive."""
    y, m = int(first[:4]), int(first[5:7])
    out: list[tuple[int, int]] = []
    while date(y, m, 1) < last_exclusive:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download_month(year: int, month: int, report: DownloadReport, *, refresh: bool = False) -> pd.DataFrame:
    tag = f"{year:04d}-{month:02d}"
    cache = RAW_ZIP_DIR / f"{SYMBOL}-{INTERVAL}-{tag}.zip"
    report.months_requested += 1

    payload = None
    if cache.exists() and not refresh:
        payload = cache.read_bytes()
    else:
        payload = _http_get(monthly_url(year, month))
        if payload is not None:
            RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(payload)

    if payload is None:
        report.months_missing.append(tag)
        print(f"  [miss] {tag}")
        return pd.DataFrame()

    df = _parse_kline_frame(_read_zip_csv(payload), f"monthly:{tag}")
    report.months_ok += 1
    report.raw_rows += len(df)
    report.malformed_rows_dropped += df.attrs.get("malformed", 0)
    report.invalid_ohlc_dropped += df.attrs.get("invalid_ohlc", 0)
    print(f"  [ok]   {tag}  {len(df):>6,} candles")
    return df


def download_day(day: date, report: DownloadReport) -> pd.DataFrame:
    tag = f"{day:%Y-%m-%d}"
    report.days_requested += 1
    payload = _http_get(daily_url(day))
    if payload is None:
        report.days_missing.append(tag)
        return pd.DataFrame()
    df = _parse_kline_frame(_read_zip_csv(payload), f"daily:{tag}")
    report.days_ok += 1
    report.raw_rows += len(df)
    report.malformed_rows_dropped += df.attrs.get("malformed", 0)
    report.invalid_ohlc_dropped += df.attrs.get("invalid_ohlc", 0)
    print(f"  [ok]   {tag}  {len(df):>6,} candles")
    return df


def download_rest_range(start_ms: int, end_ms: int, report: DownloadReport) -> pd.DataFrame:
    """Paginated REST backfill. Used for the newest candles not yet archived."""
    rows: list[list] = []
    cursor = start_ms
    guard = 0
    while cursor < end_ms and guard < 50:
        guard += 1
        batch = _http_get(
            f"{REST_BASE}/api/v3/klines",
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": REST_LIMIT,
            },
            as_json=True,
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + INTERVAL_MINUTES * 60 * 1000
        if len(batch) < REST_LIMIT:
            break
        time.sleep(0.25)                     # stay well inside rate limits

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = _parse_kline_frame(raw, "rest")
    report.rest_rows = len(df)
    report.raw_rows += len(df)
    print(f"  [ok]   REST top-up  {len(df):,} candles")
    return df


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def download_all(
    *,
    first_month: str = FIRST_DATA_MONTH,
    refresh_recent_months: int = 2,
    include_rest_topup: bool = True,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, DownloadReport]:
    """
    Build the full raw BTCUSDT 15m history and write data/raw/btcusdt_15m_raw.csv.

    `refresh_recent_months` re-downloads the most recent N monthly archives even
    if cached, because Binance occasionally republishes a month.
    """
    now = now or datetime.now(timezone.utc)
    report = DownloadReport()
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    current_month_start = date(now.year, now.month, 1)
    months = month_range(first_month, current_month_start)
    print(f"Downloading {len(months)} monthly archives ({first_month} .. before {current_month_start:%Y-%m})")

    frames: list[pd.DataFrame] = []
    for idx, (y, m) in enumerate(months):
        refresh = idx >= len(months) - refresh_recent_months
        part = download_month(y, m, report, refresh=refresh)
        if not part.empty:
            frames.append(part)

    # The current month has no monthly archive yet -> use daily archives.
    print(f"Downloading daily archives for open month {current_month_start:%Y-%m}")
    day = current_month_start
    yesterday = now.date() - timedelta(days=1)
    while day <= yesterday:
        part = download_day(day, report)
        if not part.empty:
            frames.append(part)
        day += timedelta(days=1)

    if not frames:
        raise RuntimeError(
            "No Binance data could be downloaded. Check network access to "
            "data.binance.vision / api.binance.com."
        )

    combined = pd.concat(frames, ignore_index=True)

    # REST top-up for candles newer than the last archived one.
    if include_rest_topup:
        latest_ms = int(combined["timestamp"].max().timestamp() * 1000)
        expected_last = int(now.timestamp() * 1000)
        if expected_last - latest_ms > INTERVAL_MINUTES * 60 * 1000:
            print("Downloading REST top-up for the newest candles")
            rest = download_rest_range(latest_ms + 1, expected_last, report)
            if not rest.empty:
                combined = pd.concat([combined, rest], ignore_index=True)

    combined = combined.drop(columns=["_source"], errors="ignore")

    # De-duplicate on timestamp, keep the row that came from the best source.
    # (REST is authoritative for the newest candles; monthly archives are
    #  authoritative historically. Re-download order already puts the most
    #  recent fetch last, so keep="last" is correct.)
    n_before = len(combined)
    combined = combined.sort_values("timestamp", kind="mergesort")
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    report.duplicate_rows_dropped = n_before - len(combined)

    combined = combined.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    report.final_rows = len(combined)
    report.first_timestamp = str(combined["timestamp"].iloc[0])
    report.last_timestamp = str(combined["timestamp"].iloc[-1])

    RAW_CANDLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_CANDLES_CSV, index=False)
    print(f"\nWrote {RAW_CANDLES_CSV}  ({len(combined):,} rows)")
    print(report.render())
    return combined, report


if __name__ == "__main__":
    download_all()
