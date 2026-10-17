"""
T09, T10 -- the prediction store.

``models/predictions.db`` is the only part of this project that keeps state
across runs, and its failure modes are the quiet kind: a window counted twice,
a projection overwritten by a later refresh, a ``--no-record`` run that writes
anyway. None of them raises; all of them corrupt the track record silently.

Every test here runs against a temporary database under pytest's ``tmp_path``.
The live ``models/predictions.db`` is never written -- the one test that opens
it opens it ``mode=ro`` and only reads.

The window fixture mirrors ``src/_check_store.py``'s ``window()`` on purpose:
same anchor, same ten rows (six direction horizons plus the four the regime
head answers). That helper is the project's own description of what a refresh
writes, so the test writes what the engine writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dashboard import CANDLE_MS
from predstore import COLUMNS, PredictionStore

# 2026-09-28 15:30 UTC -- the anchor src/_check_store.py uses.
ANCHOR = 1790609400000

DIRECTION_HORIZONS = (1, 2, 3, 4, 5, 6)
REGIME_HORIZONS = (1, 2, 4, 6)                 # the regime head answers four
ROWS_PER_WINDOW = len(DIRECTION_HORIZONS) + len(REGIME_HORIZONS)


def window(anchor: int, *, pct: float = 0.1) -> list[dict]:
    """
    One window exactly as ``LiveEngine._persist()`` writes it: six direction
    rows and four regime rows, one per horizon that head answers.
    """
    rows = []
    for k in DIRECTION_HORIZONS:
        rows.append(dict(anchor_ts=anchor, horizon=f"{k * 15}m", kind="direction",
                         made_at="2026-09-28T15:30:00+00:00", model="m",
                         candles=k, target_ts=anchor + k * CANDLE_MS,
                         predicted=0, confidence=0.4, expected_pct=pct,
                         projected_close=100.0 + pct, activity=None))
    for k in REGIME_HORIZONS:
        rows.append(dict(anchor_ts=anchor, horizon=f"{k * 15}m", kind="regime",
                         made_at="2026-09-28T15:30:00+00:00", model="m",
                         candles=k, target_ts=anchor + k * CANDLE_MS,
                         predicted=1, confidence=0.7, expected_pct=0.0,
                         projected_close=100.0, activity="ACTIVE"))
    return rows


def row_count(path: Path) -> int:
    """Row count read straight out of the file, bypassing PredictionStore."""
    db = sqlite3.connect(str(path))
    try:
        return int(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    finally:
        db.close()


@pytest.fixture
def store(tmp_path):
    """A writable store in a temp directory. Nothing here touches models/."""
    s = PredictionStore(tmp_path / "predictions.db")
    yield s
    s.close()


@pytest.fixture
def filled(tmp_path):
    """A store with exactly one recorded window, plus its path."""
    path = tmp_path / "predictions.db"
    s = PredictionStore(path)
    assert s.record(window(ANCHOR)) == ROWS_PER_WINDOW
    yield s, path
    s.close()


# ---------------------------------------------------------------------------
# T09 -- recording the same window twice
# ---------------------------------------------------------------------------

def test_T09_recording_the_same_window_twice_adds_no_rows(store):
    """
    T09 -- record the exact same window twice; the second write must be a
    no-op. Expected: no duplicate rows.
    """
    first = store.record(window(ANCHOR))
    assert first == ROWS_PER_WINDOW, (
        f"the first write inserted {first} row(s), expected {ROWS_PER_WINDOW} "
        f"(six direction horizons + {len(REGIME_HORIZONS)} regime horizons)"
    )

    rows_after_first = row_count(store.path)
    second = store.record(window(ANCHOR))

    assert second == 0, (
        f"T09 FAILED: recording the same window a second time inserted "
        f"{second} row(s); the store is not idempotent.\n"
        f"  rows after the first write : {rows_after_first}\n"
        f"  rows after the second write: {row_count(store.path)}"
    )
    assert row_count(store.path) == rows_after_first, (
        f"T09 FAILED: the row count moved from {rows_after_first} to "
        f"{row_count(store.path)} even though record() reported {second} new "
        "rows"
    )

    stats = store.stats()
    assert stats["n_rows"] == ROWS_PER_WINDOW, (
        f"the store holds {stats['n_rows']} rows after two identical writes; "
        f"expected {ROWS_PER_WINDOW}"
    )
    assert stats["n_windows"] == 1, (
        f"two writes of ONE window produced {stats['n_windows']} windows"
    )


def test_T09_a_repeat_cannot_overwrite_the_first_write(store):
    """
    T09 -- and the repeat must not quietly *replace* the original either. The
    first projection for a candle is the one that was actually made at the
    time; a later refresh must not be able to rewrite history.
    """
    store.record(window(ANCHOR, pct=0.1))

    tampered = [dict(row, projected_close=999.0, confidence=0.99)
                for row in window(ANCHOR, pct=0.1)]
    inserted = store.record(tampered)

    assert inserted == 0, f"the overwrite attempt inserted {inserted} row(s)"

    db = sqlite3.connect(str(store.path))
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            "SELECT projected_close, confidence FROM predictions "
            "WHERE kind = 'direction'").fetchall()
    finally:
        db.close()

    assert len(rows) == len(DIRECTION_HORIZONS)
    for row in rows:
        assert row["projected_close"] == pytest.approx(100.1), (
            f"a recorded projection was overwritten: projected_close is "
            f"{row['projected_close']}, not the 100.1 that was first written"
        )
        assert row["confidence"] == pytest.approx(0.4), (
            f"a recorded confidence was overwritten: {row['confidence']}"
        )


def test_T09_repeating_many_times_is_still_one_window(store):
    """
    T09 -- the idempotency has to hold for a *sequence* of refreshes, not just
    one retry: the engine polls every RECORD_TICK_SECONDS and only the first
    write of a window may count.
    """
    store.record(window(ANCHOR))
    for _ in range(25):
        store.record(window(ANCHOR))

    stats = store.stats()
    assert stats["n_rows"] == ROWS_PER_WINDOW, (
        f"26 writes of one window left {stats['n_rows']} rows"
    )
    assert stats["n_windows"] == 1
    assert store.history()["n_points"] == 6, (
        f"the recorded path has {store.history()['n_points']} points; one "
        "window names six direction candles"
    )


def test_T09_the_primary_key_is_what_makes_the_write_idempotent(store):
    """
    T09 -- assert the *mechanism*, not just the outcome. ``record()`` uses
    ``INSERT OR IGNORE``, which only ignores a conflict if the table declares
    the right primary key. If someone drops the key, every check above would
    start inserting duplicates -- and this is the assertion that says so.
    """
    db = sqlite3.connect(str(store.path))
    try:
        info = db.execute("PRAGMA table_info(predictions)").fetchall()
    finally:
        db.close()

    pk_columns = [row[1] for row in sorted(
        (row for row in info if row[5]), key=lambda row: row[5])]
    assert pk_columns == ["anchor_ts", "horizon", "kind"], (
        f"the predictions table's primary key is {pk_columns}, expected "
        "['anchor_ts', 'horizon', 'kind'] - the key is what makes a repeated "
        "refresh a no-op"
    )

    declared = {row[1] for row in info}
    missing = set(COLUMNS) - declared
    assert not missing, f"the table is missing column(s) {sorted(missing)}"


def test_T09_the_two_heads_do_not_collide(store):
    """
    T09 -- the key includes ``kind``. Both heads answer the 15m, 30m, 60m and
    90m candles with the same anchor, so without ``kind`` in the key the regime
    row would displace the direction row (or be dropped as a duplicate).
    """
    store.record(window(ANCHOR))

    db = sqlite3.connect(str(store.path))
    try:
        both = db.execute(
            "SELECT COUNT(*) FROM predictions "
            "WHERE anchor_ts = ? AND horizon = '15m'", (ANCHOR,)).fetchone()[0]
    finally:
        db.close()

    assert both == 2, (
        f"anchor {ANCHOR} horizon 15m has {both} row(s); both heads answer "
        "that horizon, so there must be one direction row and one regime row"
    )


def test_T09_a_later_window_only_adds_its_own_rows(store):
    """
    T09 -- consecutive refreshes overlap: the next window's 15m target is the
    current window's 30m target. Only the genuinely new (anchor, horizon, kind)
    triples may be inserted.
    """
    first = store.record(window(ANCHOR))
    assert first == ROWS_PER_WINDOW

    # The next window's anchors are new, so all ten of its rows are new...
    second = store.record(window(ANCHOR + CANDLE_MS))
    assert second == ROWS_PER_WINDOW, (
        f"a window at a new anchor inserted {second} row(s), expected a fresh "
        f"{ROWS_PER_WINDOW}"
    )
    assert row_count(store.path) == 2 * ROWS_PER_WINDOW

    # ...and re-recording the pair changes nothing.
    assert store.record(window(ANCHOR)) == 0
    assert store.record(window(ANCHOR + CANDLE_MS)) == 0
    assert row_count(store.path) == 2 * ROWS_PER_WINDOW

    hist = store.history()
    assert hist["n_windows"] == 2, f"history counts {hist['n_windows']} windows"

    # two windows one candle apart name targets +1..+6 and +2..+7, i.e. seven
    # distinct candles -- the overlapping five are merged, not double-drawn.
    assert len(hist["points"]) == 7, (
        f"two windows one candle apart name 7 distinct direction candles, but "
        f"the recorded path has {len(hist['points'])} point(s)"
    )
    assert [p["target_ts"] for p in hist["points"]] == sorted(
        p["target_ts"] for p in hist["points"]), (
        "history must read oldest first"
    )

    # NOTE: history()["n_points"] counts recorded ROWS of this kind (12 here),
    # not distinct candles -- the UI labels it "Recorded projections: N points"
    # and pairs it with n_windows, so that is what it is. Pinned here so the
    # distinction cannot quietly change meaning.
    assert hist["n_points"] == 2 * len(DIRECTION_HORIZONS) == 12, (
        f"n_points is {hist['n_points']}; it counts recorded direction rows "
        f"(2 windows x {len(DIRECTION_HORIZONS)} horizons = 12)"
    )
    assert sum(p["n"] for p in hist["points"]) == hist["n_points"], (
        "the merged points do not add up to the recorded projections"
    )


def test_T09_the_count_returned_is_the_number_actually_written(filled):
    """
    T09 -- ``record()`` reports how many rows are new, and that number has to
    match the file, because the engine logs it as "n new calls". A report that
    disagrees with the table is a wrong number in the track record.
    """
    store, path = filled
    before = row_count(path)

    assert store.record(window(ANCHOR + 3 * CANDLE_MS)) == ROWS_PER_WINDOW, (
        "a wholly new window was not reported as ten new rows"
    )
    assert row_count(path) - before == ROWS_PER_WINDOW, (
        f"record() reported {ROWS_PER_WINDOW} new rows but the file grew by "
        f"{row_count(path) - before}"
    )

    partial = [dict(window(ANCHOR + 3 * CANDLE_MS)[0],
                    horizon="7.5m", candles=7, target_ts=ANCHOR + 7 * CANDLE_MS)]
    assert store.record(partial) == 1, "a single new row was not reported as 1"
    assert store.record(partial) == 0, "the same single row was counted twice"


# ---------------------------------------------------------------------------
# T10 -- read-only mode
# ---------------------------------------------------------------------------

def test_T10_a_read_only_store_refuses_to_write(filled):
    """
    T10 -- open the database read-only and attempt a write. Expected: the write
    is refused and the existing row count is unchanged.
    """
    _writer, path = filled
    before = row_count(path)

    ro = PredictionStore(path, create=False)
    try:
        assert ro.available, "a read-only store failed to open an existing file"

        returned = ro.record(window(ANCHOR + 100 * CANDLE_MS))
        assert returned == 0, (
            f"T10 FAILED: record() on a read-only store returned {returned}, "
            "expected 0"
        )
        assert row_count(path) == before, (
            f"T10 FAILED: the row count changed from {before} to "
            f"{row_count(path)} after a write to a read-only store"
        )

        assert ro.resolve({ANCHOR + k * CANDLE_MS: 123.0
                           for k in DIRECTION_HORIZONS}) == 0, (
            "resolve() wrote to a read-only store"
        )
        assert row_count(path) == before

        # ...and it can still read, which is the point of opening it at all.
        assert ro.history()["n_points"] > 0, (
            "a read-only store could not read the history it was opened for"
        )
        assert ro.stats()["n_rows"] == before
    finally:
        ro.close()

    assert row_count(path) == before, "the file changed after the store closed"


def test_T10_the_refusal_is_sqlites_not_only_the_soft_guard(filled):
    """
    T10 -- ``record()`` returns early when ``_writable`` is False. That guard
    is a decision inside the class; it is not what makes the file safe. The
    connection itself must be opened ``mode=ro``, so that a *later* caller who
    forgets to check the guard hits SQLite's own refusal rather than silently
    writing.
    """
    _writer, path = filled
    before = row_count(path)

    ro = PredictionStore(path, create=False)
    try:
        assert not ro._writable, (
            "PredictionStore(create=False) reports itself writable"
        )

        # Bypass record() entirely and write through the store's own handle.
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            ro._db.execute(
                "INSERT INTO predictions "
                f"({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                tuple(window(ANCHOR + 500 * CANDLE_MS)[0][c] for c in COLUMNS),
            )
        assert "readonly" in str(excinfo.value).lower(), (
            f"the write failed, but not because the database is read-only: "
            f"{excinfo.value}"
        )

        # And through a connection of our own, to prove the refusal is a
        # property of the file's open mode rather than of this object.
        raw = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                raw.execute("DELETE FROM predictions")
        finally:
            raw.close()
    finally:
        ro.close()

    assert row_count(path) == before, (
        f"the row count changed from {before} to {row_count(path)} despite "
        "every write being refused"
    )


def test_T10_a_read_only_store_creates_nothing(tmp_path):
    """
    T10 -- and the same rule for ``--no-record`` against a file that is not
    there: it must neither create an empty database nor leave a directory
    behind. A file that looks like evidence but holds none is worse than no
    file.
    """
    missing = tmp_path / "never_written" / "predictions.db"

    ro = PredictionStore(missing, create=False)
    try:
        assert not ro.available, (
            "a read-only store reported itself available for a file that does "
            "not exist"
        )
        assert not missing.exists(), f"the store created {missing}"
        assert not missing.parent.exists(), (
            f"the store created the directory {missing.parent} while opening "
            "read-only"
        )

        # Every read path must degrade to empty, not raise.
        assert ro.record(window(ANCHOR)) == 0
        assert ro.resolve({ANCHOR: 1.0}) == 0
        assert ro.history() == {"points": [], "n_points": 0, "n_windows": 0,
                               "kind": "direction"}
        stats = ro.stats()
        assert stats["n_rows"] == 0 and stats["available"] is False
    finally:
        ro.close()

    assert not missing.exists(), "something created the file on close"


def test_T10_a_file_that_is_not_a_store_is_not_an_error(tmp_path):
    """
    T10 -- read-only against a file that exists but is not a prediction store
    (an empty file, a stray JSON, a truncated download). A reader must treat it
    as "nothing to read" rather than raising into the server's startup path.
    """
    junk = tmp_path / "not-a-store.db"
    junk.write_text("this is not a SQLite database", encoding="utf-8")

    ro = PredictionStore(junk, create=False)
    try:
        assert not ro.available, (
            "a non-database file was reported as an available store"
        )
        assert ro.stats()["n_rows"] == 0
        assert ro.history()["n_points"] == 0
        assert ro.record(window(ANCHOR)) == 0
    finally:
        ro.close()

    assert junk.read_text(encoding="utf-8") == "this is not a SQLite database", (
        "the file was modified while being opened read-only"
    )


def test_T10_the_shipped_store_is_read_only_when_opened_that_way():
    """
    T10 -- the same guarantee against the file the project actually ships.

    Read-only, so this cannot damage the real track record; it is skipped when
    ``models/predictions.db`` has not been created yet.
    """
    from _support import MODELS_DIR

    path = MODELS_DIR / "predictions.db"
    if not path.exists():
        pytest.skip(
            f"{path} does not exist - no track record has been recorded yet. "
            "Run `python src/server.py --port 8000` for a minute first."
        )

    before = row_count(path)
    ro = PredictionStore(path, create=False)
    try:
        assert ro.available, (
            "the shipped predictions.db could not be opened read-only - if it "
            "is mid-WAL-checkpoint the file may be temporarily unusable"
        )
        assert not ro._writable
        assert ro.record(window(ANCHOR)) == 0
        assert row_count(path) == before, (
            f"the shipped store changed from {before} to {row_count(path)} rows"
        )
        stats = ro.stats()
        assert stats["n_rows"] == before
        assert stats["path"] == str(path)
    finally:
        ro.close()

    assert row_count(path) == before, (
        "the shipped store changed after the read-only handle closed"
    )
