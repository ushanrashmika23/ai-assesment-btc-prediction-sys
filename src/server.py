"""
Live JSON/SSE API for the React dashboard in `web/`.

    pip install fastapi uvicorn          # optional - only this file needs them
    python src/server.py                 # API on http://127.0.0.1:8000
    python src/server.py --serve-web     # ... and host web/dist at /

Why this exists
---------------
The terminal dashboard (src/dashboard.py) is a good viewer for one person at a
terminal. A browser view needs the same numbers as JSON, and the one thing that
must not happen is the two drifting apart - a React panel computing its own
accuracy would be describing a different model from the one the scorecard
describes.

So this file contains NO prediction, scoring or labelling logic. It drives the
SAME `Dashboard` object the terminal UI drives (`refresh()` fetches candles,
builds features through the training path, runs the ensemble, issues calls and
resolves them) and serialises the result. Every honest caveat the terminal
version enforces - regime calls are scored against their own trailing reference,
a window spanning a data gap is never scored, each model keeps its own track
record - is inherited rather than restated.

Endpoints
---------
    GET /api/health          liveness, model name, last fetch
    GET /api/snapshot        everything the UI needs, in one round trip
    GET /api/candles?limit=  OHLCV only
    GET /api/prediction      the model read + the projected curve
    GET /api/track-record    resolved calls and the tallies
    GET /api/metrics         held-out metrics from models/eval_summary.json
    GET /api/stream          the snapshot again, pushed as Server-Sent Events

The record runs whether or not a browser is watching
----------------------------------------------------
A refresh is what writes a projection down, and a refresh used to happen only
because a request arrived. So closing the tab - or merely backgrounding it,
where the browser throttles the stream and the timers - quietly stopped the
record and left a gap in the chart. `LiveEngine.start_recorder()` takes that
decision away from the client: for as long as this process is up, the model
runs and the projection is stored on the server's own cadence. The browser is
a viewer of the record, never the reason it exists.

The projected curve, and what it is not
---------------------------------------
`prediction.curve` projects the close forward over the next six candles using
the regression head's per-horizon expected return. It is drawn because a
"predicted price path" is what the model literally outputs - NOT because it is
a forecast. On this dataset the regression head is worse than predicting no
change at every horizon (models/train_metrics.json, measured on held-out test
data), so the API also returns `naive_curve`: the same curve with zero return.
The UI draws both, and they are nearly on top of each other, which is the
honest thing for the chart to show.

`prediction.band` is the middle 50% of REALISED h-candle returns on the TRAIN
split, conditional on the direction class the model predicts at that horizon.
It is a description of the training distribution, not a confidence interval for
this particular path, and the UI labels it that way.

This is a research viewer. It places no orders, sizes no positions and models
no fees or slippage. Nothing here is investment advice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (                                            # noqa: E402
    FINAL_DATASET_CSV,
    FINAL_STARTS_TRAIN,
    HORIZON_LABELS,
    INFERENCE_BUNDLE,
    INTERVAL,
    LOOKBACK,
    MODELS_DIR,
    REGIME_CLASSES,
    SYMBOL,
)
from dashboard import (                                         # noqa: E402
    CANDLES_BY_LABEL,
    CANDLE_MS,
    HORIZON_ORDER,
    Dashboard,
    to_epoch_ms,
)
from inference import load_metadata, predict, predict_proba     # noqa: E402
from predstore import PredictionStore                            # noqa: E402

# Every projection is written here. A database rather than a JSON file because
# the chart reads it back on every refresh, and rewriting a growing JSON blob
# per refresh is exactly the kind of thing that works until it does not.
PREDICTIONS_DB = MODELS_DIR / "predictions.db"

# A break in the recorded curve longer than this many candles is drawn as a
# break, not as a straight line across it. A gap now means the server itself was
# down (`LiveEngine.start_recorder()` keeps recording while it is up, whether or
# not a browser is attached), and a line spanning that would imply the model
# said something over the outage when it said nothing.
HISTORY_GAP_CANDLES = 4

# How many points of recorded history the payload carries.
HISTORY_POINTS = 600

# Middle 50% + tails of the realised-return distribution, per horizon.
BAND_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CURVE_BANDS_JSON = MODELS_DIR / "curve_bands.json"

# Histogram geometry for the return-distribution chart, in percent.
HIST_BINS = 60
HIST_RANGE_PCT = 4.0

# How long a fetched snapshot is reused for. Binance publishes a new 15m candle
# every 15 minutes, so this only bounds how fast the UI notices a new one; it
# is the SSE push interval, not a prediction interval.
CACHE_SECONDS = 12.0

# How often the recorder re-runs the model with no browser asking. A candle is
# 15 minutes, so this only has to be comfortably under that to catch every one;
# 30s means a candle is projected within half a minute of opening, and a page
# whose tab was throttled, backgrounded or closed does not cost the record a
# single point.
RECORD_TICK_SECONDS = 30.0


# ---------------------------------------------------------------------------
# The engine: a Dashboard, driven headlessly
# ---------------------------------------------------------------------------

class LiveEngine:
    """
    Owns the Dashboard, the model and the caches, behind one lock.

    Every endpoint reads `snapshot()`, which either returns the cached payload
    or refreshes the Dashboard first. `threading.Lock` matters because the
    model is CPU-bound: two browser tabs polling at once must not both run
    Keras on the same arrays, and a refresh must not be observed half-applied.
    The recorder thread takes the same lock, so it is one more caller and not
    a second writer.
    """

    def __init__(self, model: str | None = None, record: bool = True,
                 record_interval: float = RECORD_TICK_SECONDS) -> None:
        self.lock = threading.RLock()
        self.record = record
        self.record_interval = record_interval
        self.started = time.time()
        self.last_error: str | None = None
        self.model_ready = False
        self._payload: dict | None = None
        self._payload_at = 0.0
        self._recorder: threading.Thread | None = None
        self._stop = threading.Event()

        args = SimpleNamespace(
            model=model,
            interval=30,
            min_confidence=0.0,
            once=not record,          # --once is Dashboard's "do not log" flag
        )
        self.dash = Dashboard(args)
        self.meta = self.dash.meta
        # Opens the file only if it already exists when we have been told not
        # to record: reading history is harmless, leaving an empty database
        # behind for a run that was never going to fill it is not.
        self.store = PredictionStore(PREDICTIONS_DB, create=record)

    # -- lifecycle ----------------------------------------------------------

    def warm(self) -> None:
        """Load the model and take the first reading. Blocking; call once."""
        with self.lock:
            self._load_model()
            self._refresh_locked()

    def start_recorder(self) -> None:
        """
        Refresh and record on the server's own clock, not the browser's.

        Recording is a side effect of refreshing: `snapshot()` re-reads when the
        cached payload is stale, and `_refresh_locked()` is what writes the row.
        So a server nobody is asking - tab closed, machine asleep, page merely
        backgrounded while the browser throttles its EventSource and timers -
        used to predict nothing and record nothing, and the chart grew a gap.

        A gap is honest when nothing was running. But while this process is up,
        the model ran and the record should say so, whether or not a browser is
        there to watch. This thread is that: it holds to the cadence on its own,
        so "server is up" and "the curve is unbroken" are the same statement.

        The tick is only worth doing when there is something to persist, and it
        is a no-op when there is not - `_persist()` writes nothing for an errored
        payload and `record()` ignores a window already written - so re-running
        repeatedly inside one candle is idempotent, not a double count.
        """
        if not self.record or self._recorder is not None:
            return
        self._stop.clear()
        self._recorder = threading.Thread(
            target=self._record_loop, daemon=True, name="btc-recorder")
        self._recorder.start()

    def stop_recorder(self) -> None:
        """Ask the recorder to finish. Safe to call when it never started."""
        self._stop.set()
        thread, self._recorder = self._recorder, None
        if thread is not None:
            thread.join(timeout=5.0)

    def _record_loop(self) -> None:
        """
        The recorder body. Never raises out: a background thread that dies on
        one bad fetch would silently stop recording for the rest of the run,
        which is the exact failure this whole mechanism exists to prevent.
        """
        while not self._stop.is_set():
            try:
                # Asking for the interval as the max age, rather than forcing:
                # the loop waits that long between ticks, so this refreshes
                # every time - but when a browser has just refreshed a second
                # ago the tick reuses that payload instead of running the
                # ensemble a second time for the same answer.
                self.snapshot(self.record_interval)
            except Exception as exc:                            # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[recorder] refresh failed: {self.last_error}",
                      flush=True)
            self._stop.wait(self.record_interval)

    def _load_model(self) -> None:
        self.dash._ensure_model()                              # noqa: SLF001
        self.model_ready = self.dash.model is not None

    def _refresh_locked(self) -> None:
        """refresh() + build the payload. Caller holds the lock."""
        self.dash.refresh()
        if self.dash.error:
            self.last_error = self.dash.error
        else:
            self.last_error = None
        payload = self._build_payload()
        # Written from the payload that is about to be served, never from a
        # second copy of the same arithmetic - a store that can disagree with
        # the API is a store nobody can debug.
        self._persist(payload)
        self._payload = payload
        self._payload_at = time.time()

    def _persist(self, payload: dict) -> None:
        """
        Save this window's projections, and settle up any whose candle is now
        in the frame. Silent when the payload is an error or recording is off.
        """
        if not self.record or not payload.get("ok"):
            return
        try:
            rows = []
            anchor_ms = int(payload["anchor"]["timestamp_ms"])
            base_close = float(payload["anchor"]["close"])
            made_at = payload["generated_at"]
            model = payload.get("model") or "unknown"
            regime_horizons = set(payload.get("regime", {}).get("horizons") or [])
            for h in payload["prediction"]["horizons"]:
                common = {
                    "anchor_ts": anchor_ms,
                    "horizon": h["label"],
                    "made_at": made_at,
                    "model": model,
                    "candles": int(h["candles"]),
                    "target_ts": anchor_ms + int(h["candles"]) * CANDLE_MS,
                    "confidence": float(h["direction_confidence"]),
                    "projected_close": float(h["projected_close"]),
                    "activity": h.get("activity"),
                }
                probs = h["direction_probs"]
                named = max(probs, key=probs.get)
                rows.append(dict(
                    common, kind="direction",
                    predicted=["DOWN", "NEUTRAL", "UP"].index(named),
                    expected_pct=float(h["expected_return_pct"]),
                ))
                # The activity head answers only some horizons, and predicts a
                # class rather than a return - stored with expected_pct 0 so a
                # reader can tell the two apart without consulting the docs.
                if h["label"] in regime_horizons and h.get("activity"):
                    # Its own confidence, not the direction head's that came in
                    # with `common`: the two heads are scored against different
                    # baselines, and a row labelled "regime" carrying a
                    # direction number is a row nobody can use.
                    rows.append(dict(
                        common, kind="regime",
                        predicted=1 if h["activity"] == "ACTIVE" else 0,
                        confidence=float(h["activity_confidence"]),
                        expected_pct=0.0,
                        projected_close=base_close,
                    ))
            n_new = self.store.record(rows)
            closes = {c["time"] * 1000: c["close"] for c in payload["candles"]}
            n_settled = self.store.resolve(closes)
            if n_new or n_settled:
                print(f"[store] +{n_new} row(s) for "
                      f"{payload['anchor']['timestamp']}, "
                      f"{n_settled} target(s) settled", flush=True)
        except Exception as exc:                                # noqa: BLE001
            # Never let bookkeeping take the dashboard down: the live view is
            # the product, the store is the record of it.
            print(f"[store] could not persist: {type(exc).__name__}: {exc}",
                  flush=True)

    def snapshot(self, max_age: float = CACHE_SECONDS) -> dict:
        with self.lock:
            if not self.model_ready:
                self._load_model()
            if self._payload is None or (time.time() - self._payload_at) > max_age:
                self._refresh_locked()
            return self._payload or {}

    # -- payload ------------------------------------------------------------

    def _build_payload(self) -> dict:
        d = self.dash
        if d.live is None or d.probs is None:
            return {
                "ok": False,
                "error": d.error or "no data yet",
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "generated_at": _now_iso(),
            }

        anchor_ms = int(to_epoch_ms([d.live.last_timestamp])[0])
        base_close = float(d.live.last_close)
        preds = predict(d.model, d.live, min_confidence=d.min_conf)

        horizons = []
        for i, p in enumerate(preds):
            candles = CANDLES_BY_LABEL[p.horizon]
            r = float(d.returns[i])
            horizons.append({
                "label": p.horizon,
                "candles": candles,
                "minutes": candles * 15,
                "direction": p.trend,
                "direction_probs": {
                    name: float(d.probs[i][k])
                    for k, name in ((0, "DOWN"), (1, "NEUTRAL"), (2, "UP"))
                },
                "direction_confidence": float(p.confidence),
                "expected_return_pct": r,
                "projected_close": base_close * (1.0 + r / 100.0),
                "activity": p.activity or None,
                "activity_confidence": float(p.activity_confidence),
                "activity_probs": self._activity_probs(p.horizon),
                "band": self._band(p.horizon, p.trend, base_close),
            })

        return {
            "ok": True,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "generated_at": _now_iso(),
            "model": self.dash.model_name,
            "model_summary": self.dash.model_rows or [],
            "anchor": {
                "timestamp_ms": anchor_ms,
                "timestamp": _iso_from_ms(anchor_ms),
                "close": base_close,
                "candles_in_window": int(d.live.n_candles),
                "input_shape": list(d.live.X.shape),
            },
            "next_candle_ms": anchor_ms + CANDLE_MS,
            "candles": self._candles(240),
            "prediction": {
                "horizons": horizons,
                "curve": self._curve(horizons, base_close, anchor_ms, naive=False),
                "naive_curve": self._curve(horizons, base_close, anchor_ms, naive=True),
                "band_note": (
                    "Middle 50% of REALISED returns on the training split, "
                    "conditional on the class the model predicts - a "
                    "description of the training distribution, not a "
                    "confidence interval for this path."
                ),
                "curve_note": (
                    "The regression head is worse than predicting no change "
                    "at every horizon (measured on held-out test data). The "
                    "flat dotted line is that baseline; the two nearly "
                    "coincide, which is the honest read."
                ),
                "caveat": self.dash.regression_note,
            },
            "regime": {
                "horizons": list(d.regime_horizons),
                "tau": {k: float(v) for k, v in d.regime_tau.items()},
                "reference_absmove": (float(d.ref_absmove)
                                      if d.ref_absmove else None),
                "reference_window_candles": int(d.ref_window),
            },
            # The frozen train-only thresholds, per horizon. Shipped so the
            # distribution chart can draw the NEUTRAL band the labels were
            # actually built with instead of a band invented for the chart.
            "thresholds": {
                h: {"lower_pct": float(lo), "upper_pct": float(hi)}
                for h, (lo, hi) in d.thresholds.items()
            },
            "bands": _load_curve_bands() or None,
            "prediction_history": self._history(),
            "track_record": self._track_record(),
            "metrics": self._metrics(),
            "status": d.status,
            "error": d.error,
        }

    def _activity_probs(self, horizon: str) -> dict | None:
        d = self.dash
        if d.regime_probs is None or not hasattr(d, "regime_horizons"):
            return None
        if horizon not in d.regime_horizons:
            return None
        j = list(d.regime_horizons).index(horizon)
        if j >= len(d.regime_probs):
            return None
        return {name: float(d.regime_probs[j][k])
                for k, name in REGIME_CLASSES.items()}

    def _band(self, horizon: str, direction: str,
              base_close: float) -> dict | None:
        """
        Projected close at the 10/25/50/75/90th percentile of realised
        h-candle returns for this horizon and this predicted class.

        Read from models/curve_bands.json, which is built from the TRAIN split
        only - the same split rule the rest of the project obeys.
        """
        bands = _load_curve_bands()
        if not bands:
            return None
        entry = (bands.get("by_direction", {})
                 .get(horizon, {})
                 .get(direction)) or bands.get("overall", {}).get(horizon)
        if not entry:
            return None
        return {
            "percentiles": [{"q": q,
                             "return_pct": entry["q"][str(q)],
                             "close": base_close * (1.0 + entry["q"][str(q)] / 100.0)}
                            for q in BAND_QUANTILES],
            "n_train": entry.get("n"),
        }

    def _curve(self, horizons: list[dict], base_close: float,
               anchor_ms: int, *, naive: bool) -> list[dict]:
        """
        The forward path, one point per horizon.

        `naive` returns the zero-return baseline, which is what the regression
        head has to beat and does not.
        """
        pts = [{
            "time": anchor_ms // 1000,
            "close": base_close,
            "kind": "anchor",
            "label": "now",
        }]
        for h in horizons:
            r = 0.0 if naive else h["expected_return_pct"]
            pts.append({
                "time": (anchor_ms + h["candles"] * CANDLE_MS) // 1000,
                "close": base_close * (1.0 + r / 100.0),
                "kind": "naive" if naive else "model",
                "label": h["label"],
                "return_pct": r,
            })
        return pts

    def _candles(self, limit: int) -> list[dict]:
        c = self.dash.candles
        if c is None or c.empty:
            return []
        tail = c.tail(limit)
        ts = to_epoch_ms(tail["timestamp"])
        return [
            {
                "time": int(t) // 1000,
                "open": _f(row.open), "high": _f(row.high),
                "low": _f(row.low), "close": _f(row.close),
                "volume": _f(row.volume),
            }
            for t, row in zip(ts, tail.itertuples())
        ]

    def _history(self) -> dict:
        """
        Every projection the model has recorded, as one point per candle.

        Each point is the mean of the projections that named that candle - one
        to six of them, depending on how many windows were within range of it.
        Direction head only: the activity head stores no return, and averaging
        its zeros in would drag the curve towards "no change".

        Candles nobody ever predicted are absent, and the gap between two
        points more than HISTORY_GAP_CANDLES apart is emitted as a whitespace
        point so the chart BREAKS the line there. Without that the series
        draws a straight segment across a weekend and reads as a forecast the
        model never made.
        """
        hist = self.store.history(HISTORY_POINTS)
        pts: list[dict] = []
        prev_ms: int | None = None
        for p in hist["points"]:
            if (prev_ms is not None
                    and p["target_ts"] - prev_ms > HISTORY_GAP_CANDLES * CANDLE_MS):
                pts.append({"time": (prev_ms + CANDLE_MS) // 1000})
            pts.append({"time": p["target_ts"] // 1000, "value": p["close"],
                        "n": p["n"],
                        "actual": p["actual_close"]})
            prev_ms = p["target_ts"]
        stats = self.store.stats()
        return {
            "points": pts,
            "n_points": hist["n_points"],
            "n_windows": hist["n_windows"],
            "n_scored": stats["n_scored"],
            "first_target_ms": stats["first_target_ms"],
            "last_target_ms": stats["last_target_ms"],
            "gap_candles": HISTORY_GAP_CANDLES,
            "db": stats["path"],
            "note": (
                "Every projection this dashboard has recorded, one point per "
                "candle, averaged over the windows that named it. Breaks are "
                "periods when nothing was running - the model made no call "
                "there. Stored in models/predictions.db."
            ),
        }

    def _track_record(self) -> dict:
        d = self.dash
        calls = d.score.calls
        resolved = [c for c in calls if c.resolved]
        # Ordered by when a call RESOLVES, not by when it was made. `calls` is
        # in insertion order, and _issue_calls() writes the six direction
        # horizons first and then the four activity horizons, whose list
        # restarts at 15m - so in insertion order target_ts climbs to +6
        # candles and then drops back to +1, and 15m/30m/60m/90m each appear
        # twice because both heads answer them. Anything placing these on a
        # time axis needs them ascending; Lightweight Charts asserts outright
        # on markers whose times are not strictly ascending. Calls that still
        # share a candle after sorting are merged by the chart, not here,
        # because the merge is a display decision (see web/src/markers.js).
        recent = sorted(calls[-80:],
                        key=lambda c: (int(c.target_ts), c.kind, c.horizon))
        out = {
            "log_file": d.log_path.name,
            "n_issues": d.score.n_issues,
            "n_calls": len(calls),
            "n_resolved": len(resolved),
            "direction": _tally(calls, "direction"),
            "regime": _tally(calls, "regime"),
            "recent": [
                {
                    "made_at": c.made_at,
                    "horizon": c.horizon,
                    "kind": c.kind,
                    "predicted": c.predicted,
                    "predicted_label": _class_name(c),
                    "confidence": float(c.confidence),
                    "price_at_call": float(c.price_at_call),
                    "target_ts": int(c.target_ts),
                    "target_ts_iso": _iso_from_ms(int(c.target_ts)),
                    "resolved": c.resolved,
                    "realized": c.realized,
                    "realized_label": _realized_name(c),
                    "realized_pct": (float(c.realized_pct)
                                     if c.realized_pct is not None else None),
                    "hit": c.hit,
                }
                for c in recent
            ],
        }
        return out

    def _metrics(self) -> dict | None:
        """
        The held-out numbers, straight from models/eval_summary.json.

        Displayed so the UI can show what the model scored on data it never
        trained on, rather than only how it is doing on the last few live calls
        (which is a tiny, noisy sample).
        """
        path = MODELS_DIR / "eval_summary.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:                                       # noqa: BLE001
            return None
        entry = summary.get(self.dash.metrics_key)
        if not entry:
            return {"available_keys": sorted(summary.keys()),
                    "key": self.dash.metrics_key,
                    "note": "this model has not been through src/evaluate.py"}
        # TEST is preferred and named as such: it is the only split no decision
        # in this project was ever taken on. Falling back to val silently would
        # present a number the model was selected against as if it were held
        # out, so which one is shown is reported alongside it.
        split = entry.get("test") or entry.get("val") or {}
        return {
            "key": self.dash.metrics_key,
            "split": "test" if entry.get("test") else "val",
            "generated_utc": summary.get("generated_utc"),
            "n_samples": split.get("n_samples"),
            "direction": {
                "macro_f1": split.get("macro_F1"),
                "accuracy": split.get("accuracy"),
                "baseline_accuracy": split.get("baseline_accuracy"),
                "best_floor": split.get("best_floor_by_macro_f1"),
                "sweep": split.get("sweep"),
            },
            "regime": split.get("regime"),
            "regression": split.get("regression"),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")


def _class_name(c) -> str:
    if c.kind == "regime":
        return REGIME_CLASSES.get(c.predicted, str(c.predicted))
    return {0: "DOWN", 1: "NEUTRAL", 2: "UP"}.get(c.predicted,
                                                  str(c.predicted))


def _realized_name(c):
    if c.realized is None or c.kind != "regime":
        return None
    return REGIME_CLASSES.get(c.realized, str(c.realized))


def _tally(calls, kind: str) -> dict:
    """
    Hit rate for one head, with the baseline it has to beat.

    The two heads are never pooled: direction is a 3-class problem whose
    majority class is NEUTRAL, activity is a 2-class problem that is balanced
    by construction, so a single pooled "accuracy" would describe neither.
    """
    mine = [c for c in calls if c.kind == kind and c.resolved]
    if not mine:
        return {"n": 0, "hits": 0, "accuracy": None, "baseline": None,
                "baseline_note": None}
    hits = sum(1 for c in mine if c.hit)
    if kind == "regime":
        baseline = 0.5
        note = ("the two activity classes are 50/50 by construction on train, "
                "so 0.50 is the number to beat")
    else:
        # The majority class of the training split, counted from the calls
        # themselves is not available; the frozen metadata carries the
        # train class fractions the thresholds were built to produce.
        baseline = None
        note = None
    return {
        "n": len(mine),
        "hits": hits,
        "accuracy": hits / len(mine),
        "baseline": baseline,
        "baseline_note": note,
        "by_horizon": _by_horizon(mine),
    }


def _by_horizon(calls) -> dict:
    out: dict[str, dict] = {}
    for c in calls:
        b = out.setdefault(c.horizon, {"n": 0, "hits": 0})
        b["n"] += 1
        b["hits"] += 1 if c.hit else 0
    for h, b in out.items():
        b["accuracy"] = b["hits"] / b["n"] if b["n"] else None
    return out


def _load_curve_bands() -> dict:
    if not CURVE_BANDS_JSON.exists():
        return {}
    try:
        with open(CURVE_BANDS_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                           # noqa: BLE001
        return {}


def build_curve_bands() -> dict:
    """
    Quantiles of realised returns, per horizon, overall and per direction
    class - computed on the TRAIN split only.

    Uses starts_train.npy to select the rows the train samples actually read
    their labels from (start + LOOKBACK - 1). Those offsets are NOT contiguous:
    gap-excluded windows leave holes, so `np.arange` would silently take rows
    from outside the train block.
    """
    import pandas as pd

    starts = np.load(FINAL_STARTS_TRAIN).astype(np.int64)
    rows = starts + LOOKBACK - 1

    cols = {h: f"future_return_{h}" for h in HORIZON_ORDER}
    labels = {h: f"direction_{h}" for h in HORIZON_ORDER}
    usecols = ["timestamp"] + list(cols.values()) + list(labels.values())
    df = pd.read_csv(FINAL_DATASET_CSV, usecols=usecols)
    sub = df.iloc[rows]

    # Histogram edges, in percent. 15m BTC returns live well inside +/-4%, and
    # everything outside the range is reported as under/overflow so the chart
    # can say so rather than silently clipping the tails.
    edges = np.linspace(-HIST_RANGE_PCT, HIST_RANGE_PCT, HIST_BINS + 1)

    out: dict = {"horizons": list(HORIZON_ORDER), "n_train_rows": int(len(rows)),
                 "hist_edges": [float(e) for e in edges],
                 "overall": {}, "by_direction": {}}
    for h in HORIZON_ORDER:
        v = sub[cols[h]].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        counts, _ = np.histogram(v, bins=edges)
        under = int((v < edges[0]).sum())
        over = int((v > edges[-1]).sum())
        out["overall"][h] = {
            "n": int(v.size),
            "q": {str(q): float(np.quantile(v, q))
                  for q in BAND_QUANTILES},
            "hist": [int(c) for c in counts],
            "hist_under": under,
            "hist_over": over,
        }
        out["by_direction"][h] = {}
        d = sub[labels[h]].to_numpy(dtype=float)
        for cls, name in ((2, "UP"), (1, "NEUTRAL"), (0, "DOWN")):
            vals = v[d == cls]
            if vals.size < 100:
                continue
            out["by_direction"][h][name] = {
                "n": int(vals.size),
                "q": {str(q): float(np.quantile(vals, q))
                      for q in BAND_QUANTILES},
            }
    with open(CURVE_BANDS_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def build_app(engine: LiveEngine, web_dist: Path | None):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    @asynccontextmanager
    async def lifespan(_app):
        # The recorder is started here rather than in main() so that it lives
        # exactly as long as the server does: up with the first request, down
        # with the shutdown, no matter how the app was built or torn down.
        engine.start_recorder()
        try:
            yield
        finally:
            engine.stop_recorder()

    app = FastAPI(
        title="BTCUSDT 15m trend API",
        description="Research viewer. Not investment advice.",
        version="1.0.0",
        lifespan=lifespan,
    )
    # The Vite dev server runs on another origin, so it needs CORS. It is
    # restricted to localhost: this API is a local research tool, and the
    # model's output should not be exposed as a public service.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        d = engine.dash
        return {
            "ok": engine.last_error is None,
            "model": d.model_name,
            "model_ready": engine.model_ready,
            "status": d.status,
            "error": d.error,
            "uptime_seconds": round(time.time() - engine.started, 1),
            "last_fetch": d.last_fetch or None,
            "bundle": INFERENCE_BUNDLE.name if INFERENCE_BUNDLE.exists() else None,
        }

    @app.get("/api/snapshot")
    def snapshot(max_age: float = Query(CACHE_SECONDS, ge=0, le=600)):
        return engine.snapshot(max_age)

    @app.get("/api/candles")
    def candles(limit: int = Query(240, ge=10, le=1000)):
        snap = engine.snapshot()
        return {"symbol": SYMBOL, "interval": INTERVAL,
                "candles": (snap.get("candles") or [])[-limit:]}

    @app.get("/api/prediction")
    def prediction():
        snap = engine.snapshot()
        return {
            "anchor": snap.get("anchor"),
            "prediction": snap.get("prediction"),
            "regime": snap.get("regime"),
            "model": snap.get("model"),
        }

    @app.get("/api/track-record")
    def track_record():
        return engine.snapshot().get("track_record") or {}

    @app.get("/api/metrics")
    def metrics():
        return engine.snapshot().get("metrics") or {}

    @app.get("/api/stream")
    def stream(max_age: float = Query(CACHE_SECONDS, ge=0, le=600)):
        """
        Server-Sent Events. One `snapshot` event per refresh, plus a comment
        heartbeat so a proxy does not close an idle connection.
        """
        def gen():
            last = None
            while True:
                try:
                    payload = engine.snapshot(max_age)
                    body = json.dumps(payload, separators=(",", ":"))
                    if body != last:          # do not repeat an identical frame
                        last = body
                        yield f"event: snapshot\ndata: {body}\n\n"
                    else:
                        yield ": keepalive\n\n"
                except Exception as exc:                    # noqa: BLE001
                    yield ("event: error\ndata: "
                           f"{json.dumps({'error': f'{type(exc).__name__}: {exc}'})}"
                           "\n\n")
                time.sleep(max(2.0, max_age))

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    if web_dist and web_dist.exists():
        from fastapi.staticfiles import StaticFiles
        # html=True so `/` serves index.html. Mounted LAST so it cannot
        # shadow /api/*.
        app.mount("/", StaticFiles(directory=str(web_dist), html=True),
                  name="web")

    return app


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live JSON API for the React dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None,
                        help="a single .keras file; default is the ensemble "
                             "named in models/inference_bundle.json")
    parser.add_argument("--serve-web", action="store_true",
                        help="also serve web/dist at / (run `npm run build` "
                             "in web/ first)")
    parser.add_argument("--no-record", action="store_true",
                        help="do not append to the track-record log")
    parser.add_argument("--record-interval", type=float,
                        default=RECORD_TICK_SECONDS, metavar="SECONDS",
                        help="how often to re-run the model and record a "
                             "projection with no browser attached "
                             f"(default {RECORD_TICK_SECONDS:g}s; must stay "
                             "under the 15m candle)")
    parser.add_argument("--rebuild-bands", action="store_true",
                        help="recompute models/curve_bands.json from the "
                             "training split and exit")
    args = parser.parse_args()

    if args.rebuild_bands:
        bands = build_curve_bands()
        print(f"wrote {CURVE_BANDS_JSON} "
              f"({bands['n_train_rows']:,} train rows)")
        return

    try:
        import uvicorn
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "The API needs fastapi and uvicorn:\n"
            "    pip install fastapi uvicorn\n"
            "(The rest of the project does not - see requirements.txt.)"
        ) from exc

    if not CURVE_BANDS_JSON.exists():
        print("building the return-distribution bands from the train split "
              "(one off) ...")
        try:
            build_curve_bands()
        except Exception as exc:                                # noqa: BLE001
            print(f"  [warn] could not build bands: {exc}")

    engine = LiveEngine(model=args.model, record=not args.no_record,
                        record_interval=max(2.0, args.record_interval))
    print(f"loading the model and taking the first reading "
          f"(TensorFlow import takes ~10s) ...")
    engine.warm()
    if engine.last_error:
        print(f"  [warn] first reading failed: {engine.last_error}")
    print(f"  model   : {engine.dash.model_name}")
    print(f"  candle  : {engine.dash.live.last_timestamp if engine.dash.live else '-'}")
    print(f"  log     : {engine.dash.log_path}")
    if engine.record:
        print(f"  recorder: every {engine.record_interval:g}s, browser or not")
    else:
        print("  recorder: off (--no-record)")

    web_dist = (_SRC.parent / "web" / "dist") if args.serve_web else None
    if args.serve_web and not (web_dist and web_dist.exists()):
        print(f"  [warn] --serve-web given but {web_dist} does not exist; "
              f"run `npm install && npm run build` in web/")

    app = build_app(engine, web_dist)
    print(f"\nAPI on http://{args.host}:{args.port}   docs at /docs")
    if web_dist and web_dist.exists():
        print(f"dashboard on http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
