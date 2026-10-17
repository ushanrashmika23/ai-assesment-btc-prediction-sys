"""
Internal: exercise models/predictions.db's behaviour against a TEMP database.

    python src/_check_store.py

The store is the one part of this project that holds state across runs, so its
failure modes are the quiet kind: a window counted twice, a projection
overwritten by a later refresh, a gap drawn as a straight line through a
weekend, a `--no-record` run that writes anyway. None of those raise. Each is
checked here, on a database in a temp directory, so the live store is never
touched and the check can be run at any time.

Stdlib only. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dashboard import CANDLE_MS                                  # noqa: E402
from predstore import PredictionStore                            # noqa: E402
from server import LiveEngine                                    # noqa: E402

ANCHOR = 1790609400000          # 2026-09-28 15:30 UTC
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<52} {detail}")
    if not ok:
        FAILURES.append(label)


def window(anchor: int, *, pct: float = 0.1) -> list[dict]:
    """One window as _persist() writes it: 6 direction rows + 4 regime rows."""
    rows = []
    for k in (1, 2, 3, 4, 5, 6):
        rows.append(dict(anchor_ts=anchor, horizon=f"{k * 15}m", kind="direction",
                         made_at="2026-09-28T15:30:00+00:00", model="m",
                         candles=k, target_ts=anchor + k * CANDLE_MS,
                         predicted=0, confidence=0.4, expected_pct=pct,
                         projected_close=100.0 + pct, activity=None))
    for k in (1, 2, 4, 6):
        rows.append(dict(anchor_ts=anchor, horizon=f"{k * 15}m", kind="regime",
                         made_at="2026-09-28T15:30:00+00:00", model="m",
                         candles=k, target_ts=anchor + k * CANDLE_MS,
                         predicted=1, confidence=0.7, expected_pct=0.0,
                         projected_close=100.0, activity="ACTIVE"))
    return rows


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / "check.db"
    store = PredictionStore(tmp)
    print(f"temp store: {tmp}\n")

    # -- writing ------------------------------------------------------------
    check("a window writes 10 rows", store.record(window(ANCHOR)) == 10)
    check("re-recording the same window adds nothing",
          store.record(window(ANCHOR)) == 0, "INSERT OR IGNORE")
    check("a later refresh cannot overwrite it",
          store.record([dict(window(ANCHOR)[0], projected_close=999.0)]) == 0
          and store.history()["points"][0]["close"] != 999.0,
          "the first write wins")
    check("a new window adds its own rows", store.record(window(ANCHOR + CANDLE_MS)) == 10)

    # Two heads answer the same candles, so 20 calls land on 7 distinct ones.
    hist = store.history()
    check("calls per candle are merged, not double-counted",
          len(hist["points"]) == 7, f"{len(hist['points'])} candle(s) from 20 calls")
    check("history counts the windows behind it",
          hist["n_windows"] == 2, f"n_windows={hist['n_windows']}")
    check("history reads oldest first",
          [p["target_ts"] for p in hist["points"]]
          == sorted(p["target_ts"] for p in hist["points"]))
    check("the direction head is what is charted",
          hist["kind"] == "direction",
          "regime rows store expected_pct=0 and would drag the mean down")

    # The regime head predicts a class, not a return. If its zeros ever leaked
    # into the mean, every point would sit below the direction-only value.
    only_dir = store.history(kind="direction")["points"][0]["close"]
    check("regime rows are excluded from the mean",
          abs(only_dir - 100.1) < 1e-9, f"mean={only_dir}")

    # -- resolving ----------------------------------------------------------
    closes = {ANCHOR + k * CANDLE_MS: 101.0 + k for k in (1, 2, 3, 4, 5, 6, 7)}
    check("settling fills the realised close", store.resolve(closes) == 7)
    check("settling again is a no-op", store.resolve(closes) == 0,
          "only outstanding targets are written")
    db = sqlite3.connect(tmp)
    check("every settled row carries a close",
          db.execute("SELECT COUNT(*) FROM predictions WHERE actual_close IS NULL "
                     "AND target_ts <= ?", (max(closes),)).fetchone()[0] == 0)

    # -- gaps ---------------------------------------------------------------
    # history() returns points; it does NOT insert the break - that is
    # _history()'s job, and _history() is what the chart is fed. Checking the
    # store here would be checking a layer that never had the behaviour.
    outage = 3 * 24 * 4                             # three days of no candles
    far = ANCHOR + outage * CANDLE_MS
    store.record(window(far, pct=0.2))
    hist = store.history()
    steps = [b["target_ts"] - a["target_ts"]
             for a, b in zip(hist["points"], hist["points"][1:])]
    # Not `== outage`: the gap is measured between the last target of the
    # window before it and the first target of the one after, and both
    # windows' horizons reach into the outage from either side. What must
    # hold is that nothing was invented in between.
    check("the store invents no points inside an outage",
          max(steps) > 250 * CANDLE_MS,
          f"largest step {max(steps) // CANDLE_MS} candles, vs 7 within a window")

    charted = LiveEngine._history(SimpleNamespace(store=store))
    pts = charted["points"]
    breaks = [i for i, p in enumerate(pts) if "value" not in p]
    check("the chart's copy breaks the line at the outage", len(breaks) == 1,
          f"{len(breaks)} whitespace point(s) - a straight line across an "
          "outage would be a forecast nobody made")
    if breaks:
        i = breaks[0]
        before, after = pts[i - 1], pts[i + 1]
        check("the break sits between two real points",
              before["time"] < pts[i]["time"] < after["time"],
              f"{before['time']} < {pts[i]['time']} < {after['time']}")
        check("the break spans the outage, not one candle",
              (after["time"] - before["time"]) > 250 * 900,
              f"spans {(after['time'] - before['time']) // 900} candles")
    check("charted times stay strictly ascending",
          [p["time"] for p in pts] == sorted({p["time"] for p in pts}),
          f"{len(pts)} point(s), {len({p['time'] for p in pts})} distinct")

    # -- the read-only guarantee -------------------------------------------
    check("--no-record against a missing file creates nothing",
          not PredictionStore(tmp.parent / "never.db", create=False).available
          and not (tmp.parent / "never.db").exists())
    ro = PredictionStore(tmp, create=False)
    check("--no-record against a real file still reads it",
          ro.available and ro.history()["n_points"] > 0)
    before = sqlite3.connect(tmp).execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    ro.record(window(ANCHOR + 999 * CANDLE_MS))
    after = sqlite3.connect(tmp).execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    check("--no-record cannot write to it", before == after,
          f"{before} row(s) before, {after} after")
    check("and the read-only mode is SQLite's, not just ours",
          not ro._writable, "opened mode=ro; a write would raise, not silently pass")

    # -- stats --------------------------------------------------------------
    st = store.stats()
    check("stats agree with the table",
          st["n_rows"] == after and st["n_windows"] == 3,
          f"{st['n_rows']} rows, {st['n_windows']} windows, {st['n_scored']} settled")

    store.close()
    ro.close()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all store checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
