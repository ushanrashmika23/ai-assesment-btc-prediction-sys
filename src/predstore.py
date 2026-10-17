"""
Every projection the dashboard makes, kept in SQLite.

Why a database next to the JSON track record rather than inside it: the JSON
log stores the CALLS - was the class right? - and is rewritten whole each time
one is issued. This stores the VALUES - the projected close at each horizon,
for every window the dashboard was ever running - so the chart can draw the
model's recorded path instead of only the projection it is making right now.
Neither is derivable from the other, and they are scored by different rules.

Gaps are expected and are NOT an error. While the server is up its recorder
thread fills every candle, so a gap now means the process itself was down -
but there are always candles nobody predicted, and readers are built to
tolerate that: history() returns the candles that exist rather than a dense
series.

Stdlib sqlite3 only - no new dependency, and the file is readable with any
SQLite client, which is the point of writing it down.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    anchor_ts       INTEGER NOT NULL,   -- ms epoch of the candle projected FROM
    horizon         TEXT    NOT NULL,   -- '15m' .. '90m'
    kind            TEXT    NOT NULL,   -- 'direction' | 'regime'
    made_at         TEXT    NOT NULL,   -- ISO UTC
    model           TEXT    NOT NULL,
    candles         INTEGER NOT NULL,   -- how far ahead, in candles
    target_ts       INTEGER NOT NULL,   -- ms epoch of the candle projected TO
    predicted       INTEGER NOT NULL,   -- class index the head chose
    confidence      REAL    NOT NULL,
    expected_pct    REAL    NOT NULL,
    projected_close REAL    NOT NULL,
    activity        TEXT,               -- regime head's ACTIVE/QUIET, else NULL
    actual_close    REAL,               -- NULL until that candle has happened
    -- One row per horizon per head per window. This is what makes re-running
    -- a refresh idempotent: INSERT OR IGNORE cannot duplicate or overwrite a
    -- window that has already been written.
    PRIMARY KEY (anchor_ts, horizon, kind)
);
CREATE INDEX IF NOT EXISTS predictions_target ON predictions(target_ts);
CREATE INDEX IF NOT EXISTS predictions_kind_target ON predictions(kind, target_ts);
"""

COLUMNS = ("anchor_ts", "horizon", "kind", "made_at", "model", "candles",
           "target_ts", "predicted", "confidence", "expected_pct",
           "projected_close", "activity")


class PredictionStore:
    """Thin, deliberately dumb wrapper over one SQLite file."""

    def __init__(self, path: str | Path, create: bool = True) -> None:
        """
        `create=False` opens an existing file READ-ONLY and does nothing at
        all if it is not there. A run that has been told not to record must
        neither leave an empty database behind nor write to a real one - the
        same rule the JSON track record follows, and for the same reason: a
        file that looks like evidence but holds none is worse than no file.

        Read-only is enforced by SQLite (`mode=ro`), not only by the guard in
        record()/resolve(). The guard is the intended behaviour; the URI is
        what makes it true even if a later caller forgets to check.
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._writable = create
        if not create and not self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The API is threaded; the engine's lock already serialises refreshes,
        # but a connection is not itself thread-safe, so take no chances.
        if create:
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            # WAL lets a reader (the payload builder) run while a write is in
            # flight, which is the normal case here: one process, two roles.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()
            return
        uri = f"file:{self.path.as_posix()}?mode=ro"
        try:
            self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            # A file that is not a store, or is a store without the table, is
            # not an error to a reader - it is simply nothing to read.
            self._db.execute("SELECT 1 FROM predictions LIMIT 1").fetchone()
        except sqlite3.Error:
            if self._db is not None:
                self._db.close()
            self._db = None

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @property
    def available(self) -> bool:
        """False when opened create=False against a file that is not there."""
        return self._db is not None

    # -- writing ------------------------------------------------------------

    def record(self, rows: list[dict]) -> int:
        """
        Insert this window's projections. Existing rows are left alone, so
        calling this twice for the same window is a no-op, not a double count.
        Returns the number of genuinely new rows.
        """
        if not rows or self._db is None or not self._writable:
            return 0
        with self._lock:
            before = self._db.total_changes
            self._db.executemany(
                f"INSERT OR IGNORE INTO predictions ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(COLUMNS))})",
                [tuple(r.get(c) for c in COLUMNS) for r in rows],
            )
            self._db.commit()
            return self._db.total_changes - before

    def resolve(self, closes: dict[int, float]) -> int:
        """
        Fill in the realised close for rows whose target candle now exists.

        `closes` maps ms epoch -> close; only the ones actually outstanding are
        written, so this stays a handful of statements per refresh rather than
        one per candle in the frame.
        """
        if self._db is None or not self._writable:
            return 0
        with self._lock:
            pending = {int(r["target_ts"]) for r in self._db.execute(
                "SELECT DISTINCT target_ts FROM predictions WHERE actual_close IS NULL")}
            hits = [(float(closes[t]), t) for t in pending if t in closes]
            if not hits:
                return 0
            self._db.executemany(
                "UPDATE predictions SET actual_close = ? "
                "WHERE target_ts = ? AND actual_close IS NULL", hits)
            self._db.commit()
            return len(hits)

    # -- reading ------------------------------------------------------------

    def history(self, limit: int = 600, kind: str = "direction") -> dict:
        """
        The recorded path, one point per target candle, oldest first.

        Averaged over every projection that named that candle - one to six of
        them, depending on how many windows were within range. Only the
        direction head is used: the regime head stores expected_pct = 0 by
        construction (it predicts a class, not a return), so folding its rows
        in would drag the line towards "no change" and make the model look
        worse than it is. That would be a bug in the chart, not a finding.

        Candles nobody predicted are simply absent. The caller decides how to
        show that - see _history_points() in server.py.
        """
        if self._db is None:
            return {"points": [], "n_points": 0, "n_windows": 0, "kind": kind}
        with self._lock:
            rows = self._db.execute(
                "SELECT target_ts, AVG(projected_close) AS close, COUNT(*) AS n, "
                "       AVG(actual_close) AS actual "
                "FROM predictions WHERE kind = ? "
                "GROUP BY target_ts ORDER BY target_ts DESC LIMIT ?",
                (kind, int(limit)),
            ).fetchall()
            n_points, n_windows = self._db.execute(
                "SELECT COUNT(*), COUNT(DISTINCT anchor_ts) FROM predictions "
                "WHERE kind = ?", (kind,)).fetchone()
        pts = [{"target_ts": int(r["target_ts"]), "close": float(r["close"]),
                "n": int(r["n"]),
                "actual_close": (None if r["actual"] is None else float(r["actual"]))}
               for r in reversed(rows)]
        return {"points": pts, "n_points": int(n_points),
                "n_windows": int(n_windows), "kind": kind}

    def stats(self) -> dict:
        if self._db is None:
            return {"n_rows": 0, "n_windows": 0, "first_target_ms": None,
                    "last_target_ms": None, "n_scored": 0, "path": str(self.path),
                    "available": False}
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n, COUNT(DISTINCT anchor_ts) AS w, "
                "       MIN(target_ts) AS first, MAX(target_ts) AS last, "
                "       SUM(actual_close IS NOT NULL) AS scored "
                "FROM predictions").fetchone()
        return {"n_rows": int(row["n"]), "n_windows": int(row["w"]),
                "first_target_ms": row["first"], "last_target_ms": row["last"],
                "n_scored": int(row["scored"] or 0), "path": str(self.path),
                "available": True}
