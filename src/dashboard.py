"""
Real-time interactive terminal dashboard for the trained trend model.

    python src/dashboard.py
    python src/dashboard.py --interval 60        # the shipped ensemble
    python src/dashboard.py --model models/gru_s42_best.keras

What it does
------------
Pulls the newest BTCUSDT 15m candles from Binance, pushes them through the
SAME cleaning / feature / scaling path the model was trained on, runs the
trained network, and renders the six horizon calls in a live terminal UI.

It also scores itself. Every call is appended to a prediction log together with
the wall-clock time at which it can be checked; once that time has passed the
dashboard looks up what the price actually did and marks the call right or
wrong. The scorecard panel is that tally - so the dashboard reports its own
measured live hit rate next to the majority-class baseline, rather than
asserting that the model works.

Controls
--------
    r   refresh now                i   cycle the refresh interval
    c   cycle the confidence floor m   cycle the available models
    l   toggle the log panel       x   clear this session's log
    q   quit

Honest limits, stated once here and shown in the UI
---------------------------------------------------
This is a research viewer, not a trading system. It places no orders, sizes no
positions and models no fees or slippage. A directional call is not a
profitability claim. The scorecard is a small live sample and will be noisy -
it is evidence, not proof. Nothing here is investment advice.

TensorFlow is required to load a model; the data path itself is TF-free and is
imported from the project's own modules so training and inference cannot drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import MISSING, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from rich.align import Align                                   # noqa: E402
from rich.console import Console, Group                        # noqa: E402
from rich.layout import Layout                                 # noqa: E402
from rich.live import Live                                     # noqa: E402
from rich.panel import Panel                                   # noqa: E402
from rich.table import Table                                   # noqa: E402
from rich.text import Text                                     # noqa: E402

from config import (                                           # noqa: E402
    DIRECTION_CLASSES,
    FINAL_METADATA,
    INFERENCE_BUNDLE,
    INTERVAL,
    LOOKBACK,
    MODELS_DIR,
    REGIME_CLASSES,
    SYMBOL,
)
from features import FEATURE_COLUMNS                           # noqa: E402
from inference import (                                        # noqa: E402
    Ensemble,
    build_live_input,
    fetch_latest_candles,
    load_model,
    resolve_model_paths,
    predict_proba,
)
from targets import classify_returns                           # noqa: E402

# The regimes the head is trained on, and its trailing reference window. Both
# are read from the frozen metadata at startup; these are the fallbacks used
# only if a build predates the regime head entirely.
REGIME_CANDLES_BY_LABEL = {"15m": 1, "30m": 2, "60m": 4, "90m": 6}

def log_path_for(model_file: str) -> Path:
    """
    One track record PER MODEL.

    Calls from two different networks must never share a scorecard: blending
    them would produce an accuracy figure that describes neither model.
    """
    return MODELS_DIR / f"dashboard_log_{Path(model_file).stem}.json"

# Candle counts for each horizon, keyed by the label the model emits.
HORIZON_ORDER = ["15m", "30m", "45m", "60m", "75m", "90m"]
CANDLES_BY_LABEL = {"15m": 1, "30m": 2, "45m": 3, "60m": 4, "75m": 5, "90m": 6}

TREND_STYLE = {"UP": "bold green", "DOWN": "bold red", "NEUTRAL": "dim yellow"}
TREND_MARK = {"UP": "^", "DOWN": "v", "NEUTRAL": "-"}

SPARK = "▁▂▃▄▅▆▇█"
CANDLE_MS = 15 * 60 * 1000


# ---------------------------------------------------------------------------
# Glyph support
# ---------------------------------------------------------------------------
# A legacy Windows console runs on a code page (cp437/cp1252) that cannot
# encode the block-drawing characters, and rich raises UnicodeEncodeError
# mid-render. Ask for UTF-8 first; if the stream still cannot carry the glyphs,
# fall back to ASCII so the dashboard renders everywhere.

def _prepare_console() -> tuple[Console, str, str, str]:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                     # noqa: BLE001
            pass

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        (SPARK + "█░").encode(encoding)
        unicode_ok = True
    except (UnicodeEncodeError, LookupError):
        unicode_ok = False

    if unicode_ok:
        return Console(legacy_windows=False), SPARK, "█", "░"
    return Console(legacy_windows=False), "_.-~=+*#", "#", "."


CONSOLE, SPARK, BAR_FULL, BAR_EMPTY = _prepare_console()


# ---------------------------------------------------------------------------
# Terminal input (non-blocking, no third-party dependency)
# ---------------------------------------------------------------------------

class Keys:
    """Non-blocking single-key reads. Windows via msvcrt, POSIX via termios."""

    def __init__(self) -> None:
        self.kind = "none"
        try:
            import msvcrt                                   # noqa: PLC0415
            self._msvcrt = msvcrt
            self.kind = "windows"
            return
        except ImportError:
            pass
        try:
            import termios                                   # noqa: PLC0415
            import tty                                       # noqa: PLC0415
            self._termios, self._tty = termios, tty
            self._fd = sys.stdin.fileno()
            self.kind = "posix"
        except ImportError:
            self.kind = "none"

    def __enter__(self):
        if self.kind == "posix":
            self._saved = self._termios.tcgetattr(self._fd)
            self._tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self.kind == "posix":
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN,
                                    self._saved)
        return False

    def get(self) -> str | None:
        if self.kind == "windows":
            if self._msvcrt.kbhit():
                try:
                    return self._msvcrt.getwch()
                except Exception:
                    return None
            return None
        if self.kind == "posix":
            import select                                    # noqa: PLC0415
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None
        return None


# ---------------------------------------------------------------------------
# Prediction log  (the dashboard's own track record)
# ---------------------------------------------------------------------------

@dataclass
class Call:
    """One recorded call for one horizon, resolved once its target time passes."""

    made_at: str                 # ISO UTC, when the call was issued
    model: str                   # which .keras file made the call
    horizon: str
    candles: int
    target_ts: int               # ms epoch of the candle that resolves it
    predicted: int               # 0 / 1 / 2 (direction) or 0 / 1 (regime)
    confidence: float
    expected_pct: float
    price_at_call: float
    realized_pct: float | None = None
    realized: int | None = None
    # "direction" (UP/NEUTRAL/DOWN) or "regime" (QUIET/ACTIVE). The two are
    # scored by different rules against different baselines, so they must never
    # be pooled into one accuracy figure.
    kind: str = "direction"
    # Regime calls only: the sample's own trailing 24h mean |15m return| at the
    # moment of the call, which is the denominator its label is defined against.
    ref_absmove: float | None = None

    @property
    def resolved(self) -> bool:
        return self.realized is not None

    @property
    def hit(self) -> bool | None:
        return None if self.realized is None else self.realized == self.predicted


@dataclass
class Scoreboard:
    """
    Session tally, and the naive baseline it has to beat.

    The baseline is the majority class of the TRAIN split (from the frozen
    metadata thresholds' own distribution) - not a number invented here.
    """

    calls: list[Call] = field(default_factory=list)
    n_issues: int = 0

    def add(self, calls: list[Call]) -> None:
        self.calls.extend(calls)
        self.n_issues += 1

    def resolved(self) -> list[Call]:
        return [c for c in self.calls if c.resolved]

    def resolve(self, closes: dict[int, float],
                thresholds: dict[int, tuple[float, float]],
                regime_tau: dict[int, float] | None = None) -> int:
        """
        Score every call whose target candle has now printed.

        Direction calls go through targets.classify_returns - the project's own
        function, with the frozen train-only thresholds - so a call's realized
        class is defined exactly the way the training labels were.
        Re-implementing the rule here could silently drift from it.

        Regime calls are scored by the same rule the training label uses:
        ACTIVE iff mean|return| over the h candles after the call exceeds the
        trailing 24h mean measured AT the call, times the frozen train median
        ratio. Both are computed from the candles actually fetched, so the
        scorecard measures the head rather than restating it.
        """
        todo = [c for c in self.calls
                if not c.resolved and c.target_ts in closes]
        if not todo:
            return 0

        direction = [c for c in todo if c.kind == "direction"]
        if direction:
            cols = [HORIZON_ORDER.index(c.horizon) for c in direction]
            y = np.full((len(direction), len(HORIZON_ORDER)), np.nan, dtype=float)
            for r, c in enumerate(direction):
                y[r, cols[r]] = (closes[c.target_ts] / c.price_at_call - 1.0) * 100.0
            labels = classify_returns(y, thresholds)      # NaN columns stay -1
            for r, c in enumerate(direction):
                c.realized_pct = float(y[r, cols[r]])
                c.realized = int(labels[r, cols[r]])

        regime = [c for c in todo if c.kind == "regime"]
        if regime:
            order = sorted(closes)                        # the candle timeline
            index = {ts: i for i, ts in enumerate(order)}
            closes_v = np.array([closes[ts] for ts in order], dtype=float)
            # |15m return| per candle, NaN on the first (no predecessor).
            absr = np.full(order.__len__(), np.nan, dtype=float)
            absr[1:] = np.abs(100.0 * (closes_v[1:] / closes_v[:-1] - 1.0))
            for c in regime:
                tau = (regime_tau or {}).get(c.horizon)
                j = index.get(c.target_ts)
                if tau is None or j is None or c.ref_absmove is None:
                    continue
                lo = j - c.candles + 1                    # candles anchor+1..anchor+h
                if lo < 1:
                    continue
                # The window must be h CONSECUTIVE candles. If the fetch had a
                # hole, these indices are not h candles apart and averaging
                # them would score the call against a window the market never
                # had - the same rule the training labels are built under.
                if order[j] - order[lo] != (c.candles - 1) * CANDLE_MS:
                    continue
                window = absr[lo:j + 1]
                if not np.isfinite(window).all():
                    continue
                c.realized_pct = float(window.mean())
                c.realized = int(window.mean() > tau * c.ref_absmove)
        return len(todo)

    def clear(self) -> None:
        self.calls.clear()
        self.n_issues = 0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def sparkline(values: np.ndarray, width: int = 0) -> str:
    """Sparkline, downsampled to `width` if the series is longer."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return ""
    if width and v.size > width:
        idx = np.linspace(0, v.size - 1, width).astype(int)
        v = v[idx]
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return SPARK[0] * v.size
    scaled = (v - lo) / (hi - lo) * (len(SPARK) - 1)
    return "".join(SPARK[int(round(s))] for s in scaled)


def bar(frac: float, width: int = 18) -> Text:
    frac = max(0.0, min(1.0, float(frac)))
    filled = int(round(frac * width))
    t = Text()
    t.append(BAR_FULL * filled, style="cyan")
    t.append(BAR_EMPTY * (width - filled), style="grey35")
    return t


def to_epoch_ms(values) -> np.ndarray:
    """
    Epoch milliseconds, whatever unit pandas happens to be storing.

    This project's timestamps are datetime64[ms, UTC] (pandas 3 keeps
    microsecond-resolution data at ms precision), but `.astype("int64")` returns
    the value in the column's OWN unit - so dividing it by a constant is a
    silent unit bug waiting to happen. That bug was real here: dividing a
    millisecond column by 10**6 produced 1970 dates, keys never matched, and
    the scorecard could never resolve a single call. `as_unit` pins the unit
    explicitly instead of assuming one.
    """
    return pd.DatetimeIndex(values).as_unit("ms").astype("int64").to_numpy()


def fmt_ts_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


def short_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%H:%M:%S")


# ---------------------------------------------------------------------------
# The dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    def __init__(self, args) -> None:
        self.args = args
        self.console = CONSOLE
        # --once is a dry run: it must not append to the track record.
        self.record = not getattr(args, "once", False)

        with open(FINAL_METADATA, encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.thresholds = {
            h: (self.meta["thresholds"]["per_horizon"][h]["lower_percent"],
                self.meta["thresholds"]["per_horizon"][h]["upper_percent"])
            for h in HORIZON_ORDER
        }
        # classify_returns() indexes thresholds by candle count, not label.
        self.thr_by_candles = {CANDLES_BY_LABEL[h]: self.thresholds[h]
                               for h in HORIZON_ORDER}

        # Resolve both sides: glob() yields absolute paths, and a user typing
        # the documented `--model models/gru_s42_best.keras` gives a relative
        # one. Comparing them unresolved silently ignored the flag and fell
        # back to the first model alphabetically.
        # Regime head settings, from the frozen metadata. Absent on a build
        # that predates it, in which case the panels simply do not show it.
        rmeta = self.meta.get("regime_targets") or {}
        self.regime_horizons = list(rmeta.get("horizons", []))
        self.regime_tau = {h: v["ratio_threshold"]
                           for h, v in (rmeta.get("thresholds") or {}).items()}
        self.regime_candles = {h: int(c) for h, c in
                               zip(rmeta.get("horizons", []),
                                   rmeta.get("horizon_candles", []))}
        self.ref_window = int(rmeta.get("reference_window_candles", 96))

        # The bundle is offered FIRST and is the default: it names the ensemble
        # the reported accuracy is about. Selecting an arbitrary member file by
        # alphabet would show live calls from a model that is one third of the
        # measured one. Individual .keras files stay selectable.
        self.bundle_members = []
        if INFERENCE_BUNDLE.exists():
            try:
                self.bundle_members = resolve_model_paths(INFERENCE_BUNDLE)
            except FileNotFoundError as exc:
                self.console.print(f"[yellow]{exc}[/yellow]")
        self.models = ([INFERENCE_BUNDLE] if self.bundle_members else [])
        self.models += sorted(p.resolve() for p in MODELS_DIR.glob("*.keras"))
        if not self.models:
            raise SystemExit(
                f"No trained model found in {MODELS_DIR}.\n"
                f"Train one first:  python src/train.py --arch gru"
            )

        if args.model:
            start = Path(args.model).resolve()
            if not start.exists():
                raise SystemExit(f"No model at {start}")
            if start not in self.models:
                self.models.append(start)   # explicit choice wins
            self.model_idx = self.models.index(start)
        else:
            self.model_idx = 0              # the bundle, or the first file
        # evaluate.py keys models/eval_summary.json by the name it evaluated
        # under, which for the ensemble is "ensemble_x3" rather than the bundle
        # filename. Looking up the wrong key would silently drop the measured
        # caveats, which is the one thing they exist to prevent.
        self.metrics_key = self._metrics_key()
        self.regression_note = self._measured_regression_note()
        self.floor_note = self._measured_floor_note()

        self.intervals = [15, 30, 60, 120, 300]
        self.interval_idx = min(
            range(len(self.intervals)),
            key=lambda i: abs(self.intervals[i] - args.interval))
        self.min_conf = args.min_confidence
        self.conf_choices = [0.0, 0.34, 0.40, 0.50, 0.60]
        self.show_log = True

        self.model = None
        self.model_name = None
        self.model_rows = None

        self.score = Scoreboard()
        self._load_log()

        self.candles: pd.DataFrame | None = None
        self.live = None
        self.returns: np.ndarray | None = None
        self.probs: np.ndarray | None = None
        self.regime_probs: np.ndarray | None = None
        self.ref_absmove: float | None = None
        self.status = "starting"
        self.error: str | None = None
        self.last_fetch: float = 0.0
        self.last_issue_ts: int | None = None
        self.started = time.time()

    def _metrics_key(self) -> str:
        path = self.models[self.model_idx]
        if path == INFERENCE_BUNDLE and self.bundle_members:
            return f"ensemble_x{len(self.bundle_members)}"
        return path.name

    def _measured_regression_note(self) -> str | None:
        """
        A one-line, MEASURED caveat about the regression head.

        Read from models/train_metrics.json, which src/train.py wrote from the
        real test evaluation. If the head's MAE is worse than predicting the
        mean, the 'expected' column is not a forecast and the panel says so
        instead of presenting the number as if it carried information.
        """
        path = MODELS_DIR / "train_metrics.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                m = json.load(fh)
            got = m["test_regression"]["MAE_%"]
            base = m["test_mae_naive_zero"]
            worse = [h for h in got if got[h] > base[h]]
            if not worse:
                return ("regression head beat the zero-return baseline on MAE "
                        "at every horizon (measured on the held-out test set)")
            if len(worse) == len(got):
                return ("MEASURED: the regression head is WORSE than predicting "
                        "the mean at every horizon - read 'expected' as an "
                        "output, not a forecast. The direction head is the part "
                        "that carries the signal.")
            return (f"MEASURED: regression head worse than predicting the mean "
                    f"at {len(worse)}/{len(got)} horizons")
        except Exception:                                     # noqa: BLE001
            return None

    def _measured_floor_note(self) -> str | None:
        """
        A one-line, MEASURED caveat about the confidence floor.

        Read from models/eval_summary.json, which src/evaluate.py writes. The
        floor is a control this dashboard offers, so it should say what the
        control does rather than let you assume it helps. On the model trained
        here it does not: downgrading weak calls to NEUTRAL costs macro-F1
        monotonically and buys essentially no accuracy, because DOWN and UP
        recall are already the scarce resource.

        Returns None when nothing has been evaluated yet - an absent
        measurement is not a clean bill of health.
        """
        path = MODELS_DIR / "eval_summary.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
            entry = s.get(self.metrics_key) or {}
            ref = entry.get("val") or entry.get("test")
            if not ref:
                return None
            sweeps = ref.get("sweep") or {}
            if not sweeps:
                return None
            # Floors other than 0.00, and whether any of them beats 0.00.
            base = sweeps.get("0.00", {}).get("macro_F1")
            if base is None:
                return None
            raised = {k: v["macro_F1"] for k, v in sweeps.items()
                      if k != "0.00"}
            if not raised or max(raised.values()) <= base:
                return (f"MEASURED: raising this floor never improved macro-F1 "
                        f"(evaluate.py swept it) - it costs signal. Leave it "
                        f"at 0.00 unless you are testing that claim.")
            best = max(raised, key=raised.get)
            return (f"MEASURED: on {'validation' if 'val' in entry else 'test'}"
                    f", a floor of {best} scored macro-F1 {raised[best]:.4f} vs "
                    f"{base:.4f} at 0.00 - see evaluate.py.")
        except Exception:                                     # noqa: BLE001
            return None

    # -- lifecycle ----------------------------------------------------------

    def _load_log(self) -> None:
        """
        Re-load the selected model's previous track record, so the scorecard
        accumulates across restarts.

        Everything is re-validated on the way in: a hand-edited, truncated or
        older-schema file should degrade to "no history", never crash the UI.
        """
        self.score.clear()
        path = log_path_for(self.models[self.model_idx].name)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            fields = Call.__dataclass_fields__
            valid = {f for f in fields}
            # Only the fields with NO default are required. New optional fields
            # (the regime head's `kind` / `ref_absmove`) must not invalidate a
            # track record written by an older build - discarding the history
            # would be a silent loss of the only live evidence there is.
            required = {n for n, f in fields.items()
                        if f.default is MISSING
                        and f.default_factory is MISSING}
            for d in raw.get("calls", []):
                kept = {k: v for k, v in d.items() if k in valid}
                if required <= set(kept):
                    self.score.calls.append(Call(**kept))
        except Exception as exc:                              # noqa: BLE001
            self.console.print(
                f"[yellow]Could not read {path.name} ({exc}); "
                f"starting a fresh log.[/yellow]")

    def _save_log(self) -> None:
        if not self.record:
            return
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = self.log_path
            payload = {
                "model": self.model_name,
                "note": ("Calls recorded live by src/dashboard.py. Realized "
                         "values are measured from actual Binance candles "
                         "using the frozen train-only thresholds - never "
                         "simulated, never back-filled."),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "calls": [vars(c) for c in self.score.calls],
            }
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            tmp.replace(path)              # atomic: never leave a half-written log
        except Exception:                                     # noqa: BLE001
            pass                            # a full disk must not kill the UI

    @property
    def log_path(self) -> Path:
        return log_path_for(self.model_name or self.models[self.model_idx].name)

    def _ensure_model(self) -> None:
        """(Re)load the selected model. Only pays the cost when it changes."""
        path = self.models[self.model_idx]
        if self.model_name == path.name:
            return
        self.metrics_key = self._metrics_key()

        if path == INFERENCE_BUNDLE:
            self.status = f"loading ensemble ({len(self.bundle_members)} members)"
            # compile=False inside load_model: the direction head's weighted
            # loss is a locally-defined class Keras cannot deserialise.
            self.model = Ensemble([load_model(p) for p in self.bundle_members])
            self.model_name = path.name
            self._load_log()
            self.model_rows = [
                ("architecture", f"ENSEMBLE of {len(self.bundle_members)}"),
                ("members", ", ".join(p.stem.replace("_best", "")
                                      for p in self.bundle_members)),
                ("combine", "mean of per-member softmax"),
                ("parameters", f"{self.model.count_params():,} "
                               f"(per member)"),
                ("lookback", f"{LOOKBACK} candles ({LOOKBACK * 15 // 60}h)"),
                ("features", str(len(FEATURE_COLUMNS))),
                ("bundle", INFERENCE_BUNDLE.name),
            ]
            return

        self.status = f"loading {path.name}"
        self.model = load_model(path)
        self.model_name = path.name
        self._load_log()            # each model has its own track record
        self.model_rows = [
            ("architecture", path.stem.replace("_multitask", "").upper()),
            ("parameters", f"{self.model.count_params():,}"),
            ("lookback", f"{LOOKBACK} candles ({LOOKBACK * 15 // 60}h)"),
            ("features", str(len(FEATURE_COLUMNS))),
            ("file", f"{path.name} ({path.stat().st_size / 1024:.0f} KB)"),
        ]

    # -- data ---------------------------------------------------------------

    def refresh(self) -> None:
        self.status = "fetching candles"
        self.error = None
        try:
            raw = fetch_latest_candles()
            self.status = "building features"
            # build_live_input re-derives the window from raw candles through
            # the project's own cleaning/feature/scale path.
            self.live = build_live_input(raw)
            self.status = "running the model"
            self.returns, self.probs, self.regime_probs = predict_proba(
                self.model, self.live)

            from preprocessing import clean_candles                 # noqa: PLC0415
            clean = clean_candles(raw, verbose=False).sort_values("timestamp")
            self.candles = clean
            closes = dict(zip(to_epoch_ms(clean["timestamp"]),
                              clean["close"].astype(float)))
            self.ref_absmove = self._trailing_absmove(clean)
            self._issue_calls()
            # Issuing first, then resolving once, also catches a call whose
            # target candle was already in this fetch (short horizons at an
            # infrequent refresh interval).
            filled = self.score.resolve(closes, self.thr_by_candles,
                                        self.regime_tau)
            if filled:
                self._save_log()

            self.last_fetch = time.time()
            self.status = "live"
        except Exception as exc:                                # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"

    def _trailing_absmove(self, clean: pd.DataFrame) -> float | None:
        """
        Mean |15m return| over the last `ref_window` candles - the sample's own
        activity reference, defined exactly as the training label defines it.
        """
        if len(clean) <= self.ref_window:
            return None
        closes = clean["close"].astype(float).to_numpy()
        tail = closes[-(self.ref_window + 1):]
        absr = np.abs(100.0 * (tail[1:] / tail[:-1] - 1.0))
        return float(absr.mean()) if np.isfinite(absr).all() else None

    def _issue_calls(self) -> None:
        """
        Record this window's calls, once per new candle.

        Guarded on the newest candle's timestamp so a 15-second refresh does not
        append the same prediction twelve times and inflate the scorecard.
        """
        if self.live is None or self.probs is None or not self.record:
            return
        anchor = int(to_epoch_ms([self.live.last_timestamp])[0])
        if self.last_issue_ts == anchor:
            return
        self.last_issue_ts = anchor

        made_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        price = float(self.live.last_close)
        fresh: list[Call] = []
        for i, h in enumerate(HORIZON_ORDER):
            cls = int(np.argmax(self.probs[i]))
            candles = CANDLES_BY_LABEL[h]
            target = anchor + candles * CANDLE_MS
            call = Call(
                made_at=made_at, model=self.model_name,
                horizon=h, candles=candles,
                target_ts=int(target), predicted=cls,
                confidence=float(self.probs[i][cls]),
                expected_pct=float(self.returns[i]),
                price_at_call=price,
            )
            fresh.append(call)

        # The activity calls, recorded on the same footing as the directional
        # ones. They need a trailing reference to be scorable at all, so a call
        # is only issued when one could be computed.
        if (self.regime_probs is not None and self.ref_absmove
                and len(self.regime_probs) == len(self.regime_horizons)):
            for j, h in enumerate(self.regime_horizons):
                candles = self.regime_candles.get(
                    h, REGIME_CANDLES_BY_LABEL.get(h, 0))
                if not candles or h not in self.regime_tau:
                    continue
                cls = int(np.argmax(self.regime_probs[j]))
                fresh.append(Call(
                    made_at=made_at, model=self.model_name,
                    horizon=h, candles=candles,
                    target_ts=int(anchor + candles * CANDLE_MS),
                    predicted=cls,
                    confidence=float(self.regime_probs[j][cls]),
                    expected_pct=0.0,      # the head predicts a class, not a return
                    price_at_call=price, kind="regime",
                    ref_absmove=self.ref_absmove,
                ))
        self.score.add(fresh)
        self._save_log()

    # -- panels -------------------------------------------------------------

    def header(self) -> Panel:
        if self.live is None:
            body = Text("waiting for the first candle fetch ...", style="dim")
            return Panel(body, title=f"{SYMBOL} {INTERVAL}", border_style="cyan")

        close = self.live.last_close
        prev = None
        if self.candles is not None and len(self.candles) > 1:
            prev = float(self.candles["close"].iloc[-2])
        chg = ((close / prev - 1) * 100) if prev else 0.0
        style = "green" if chg >= 0 else "red"

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("last candle",
                  fmt_ts_ms(int(to_epoch_ms([self.live.last_timestamp])[0])) + " UTC",
                  "close", Text(f"{close:,.2f}", style="bold"))
        t.add_row("change", Text(f"{chg:+.2f}%", style=style),
                  "candles in window", f"{self.live.n_candles:,}")
        age = time.time() - self.last_fetch if self.last_fetch else 0
        t.add_row("next refresh", f"{max(0, self.intervals[self.interval_idx] - age):.0f}s",
                  "confidence floor", f"{self.min_conf:.2f}")
        return Panel(t, title=f"[bold]{SYMBOL} {INTERVAL} - live trend read[/bold]",
                     subtitle=f"[dim]model {self.model_name}[/dim]",
                     border_style="cyan")

    def price_panel(self) -> Panel:
        if self.candles is None or self.candles.empty:
            return Panel(Text("no candles yet", style="dim"), title="PRICE")
        tail = self.candles.tail(LOOKBACK)
        closes = tail["close"].to_numpy(dtype=float)
        hi, lo = float(closes.max()), float(closes.min())
        first, last = float(closes[0]), float(closes[-1])
        move = (last / first - 1) * 100
        style = "green" if move >= 0 else "red"

        spark = sparkline(closes, width=min(88, self.console.width - 24))
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", justify="right")
        g.add_column()
        g.add_row("last 16h", Text(spark, style=style))
        g.add_row("range", f"{lo:,.2f} - {hi:,.2f}   "
                           f"({(hi / lo - 1) * 100:.2f}% wide)   "
                           f"net {move:+.2f}%")
        return Panel(g, title="PRICE", border_style="blue")

    def prediction_panel(self) -> Panel:
        if self.probs is None:
            return Panel(Text("no prediction yet", style="dim"),
                         title="PREDICTION", border_style="magenta")

        t = Table(box=None, pad_edge=False, expand=False)
        t.add_column("horizon", style="bold", justify="left", width=7)
        t.add_column("call", justify="left", width=9)
        t.add_column("conf", justify="right", width=6)
        t.add_column("", width=20)
        t.add_column("activity", justify="left", width=14)
        t.add_column("expected", justify="right", width=10)
        t.add_column("p(DOWN/NEU/UP)", justify="left", style="dim")

        # The activity read is the headline, so it goes on every row the head
        # covers rather than in a separate panel the eye skips.
        regime_at = {h: j for j, h in enumerate(self.regime_horizons)}
        for i, h in enumerate(HORIZON_ORDER):
            p = self.probs[i]
            cls = int(np.argmax(p))
            conf = float(p[cls])
            shown = cls
            downgraded = conf < self.min_conf
            if downgraded:
                shown = 1                              # NEUTRAL = the no-trade zone
            trend = DIRECTION_CLASSES[shown]
            label = Text(f"{TREND_MARK[trend]} {trend}", style=TREND_STYLE[trend])
            if downgraded:
                label.append(" *", style="dim")

            activity = Text("-", style="dim")
            j = regime_at.get(h)
            if (j is not None and self.regime_probs is not None
                    and j < len(self.regime_probs)):
                rp = self.regime_probs[j]
                rcls = int(np.argmax(rp))
                name = REGIME_CLASSES[rcls]
                activity = Text(f"{name} {float(rp[rcls]):.0%}",
                                style=("bold cyan" if name == "ACTIVE"
                                       else "dim"))
            t.add_row(
                h, label, f"{conf:.1%}", bar(conf), activity,
                Text(f"{self.returns[i]:+.3f}%",
                     style="green" if self.returns[i] >= 0 else "red"),
                f"{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}",
            )

        sub = ("trend label is argmax of the direction head; "
               "expected is the regression head's percent return; "
               "activity is the regime head")
        if self.min_conf > 0:
            sub += f";  * = below the {self.min_conf:.2f} floor, shown as NEUTRAL"
        body = [t, Text(sub, style="dim")]
        if self.regime_horizons and self.ref_absmove:
            body.append(Text(
                f"activity = will the next window be busier than this "
                f"sample's own trailing 24h?  (reference now "
                f"{self.ref_absmove:.3f}% per 15m candle; classes are 50/50 by "
                f"construction, so 0.50 is the baseline)", style="dim"))
        if self.regression_note:
            body.append(Text(self.regression_note, style="yellow"))
        if self.min_conf > 0 and self.floor_note:
            body.append(Text(self.floor_note, style="yellow"))
        return Panel(Group(*body),
                     title="PREDICTION - 6 horizons", border_style="magenta")

    def score_panel(self) -> Panel:
        resolved = self.score.resolved()
        n = len(resolved)
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", justify="right")
        g.add_column()

        if n == 0:
            g.add_row("calls issued", f"{len(self.score.calls)} "
                                      f"({self.score.n_issues} window(s))")
            g.add_row("resolved", "0 - the first calls resolve once their "
                                  "horizon elapses")
            return Panel(g, title="LIVE SCORECARD", border_style="yellow",
                         subtitle="[dim]measured, not asserted[/dim]")

        hits = sum(1 for c in resolved if c.hit)
        acc = hits / n

        # Split the two heads before scoring anything. A regime call is right
        # half the time by construction, so pooling it with directional calls
        # would move the headline toward 0.50 without either head changing.
        directional = [c for c in resolved if c.kind == "direction"]
        regime = [c for c in resolved if c.kind == "regime"]
        if directional and regime:
            n = len(directional)
            hits = sum(1 for c in directional if c.hit)
            acc = hits / n

        # Majority-class baseline, taken from the frozen train distribution
        # rather than assumed: the thresholds put NEUTRAL at 40% of train.
        majority = self.meta["thresholds"].get("neutral_quantile")
        base = float(majority) if majority else None

        # Per-horizon, so a single strong horizon cannot hide five weak ones.
        t = Table(box=None, pad_edge=False)
        t.add_column("horizon", style="bold", width=7)
        t.add_column("resolved", justify="right", width=8)
        t.add_column("hit", justify="right", width=5)
        t.add_column("accuracy", justify="right", width=8)
        t.add_column("", width=20)
        t.add_column("mean realized", justify="right", width=13)
        for h in HORIZON_ORDER:
            sub = [c for c in directional if c.horizon == h]
            if not sub:
                continue
            hh = sum(1 for c in sub if c.hit)
            a = hh / len(sub)
            realized = float(np.mean([c.realized_pct for c in sub]))
            t.add_row(h, str(len(sub)), str(hh), f"{a:.1%}", bar(a),
                      Text(f"{realized:+.3f}%",
                           style="green" if realized >= 0 else "red"))

        g.add_row("overall", Text(f"{hits}/{n} = {acc:.1%}",
                                  style="bold"))
        if base is not None:
            delta = acc - base
            g.add_row(
                "vs majority class",
                Text(f"{base:.1%} baseline   ({delta:+.1%})",
                     style="green" if delta > 0 else "red"),
            )
        g.add_row("mean confidence",
                  f"{np.mean([c.confidence for c in directional]):.1%}  "
                  f"(overconfidence shows here)")
        g.add_row("issues", f"{self.score.n_issues} window(s), "
                            f"{len(self.score.calls)} calls")

        # The activity head, scored separately because it is a different
        # question with a different baseline. 0.50 is not a guess here: the
        # threshold is the train median of a self-normalising ratio, so the two
        # classes are balanced by construction and ~0.50 IS the floor.
        if regime:
            rh = sum(1 for c in regime if c.hit)
            ra = rh / len(regime)
            rt = Table(box=None, pad_edge=False)
            rt.add_column("horizon", style="bold", width=7)
            rt.add_column("resolved", justify="right", width=8)
            rt.add_column("hit", justify="right", width=5)
            rt.add_column("accuracy", justify="right", width=8)
            rt.add_column("", width=20)
            rt.add_column("", width=13)
            for h in HORIZON_ORDER:
                sub = [c for c in regime if c.horizon == h]
                if not sub:
                    continue
                hh = sum(1 for c in sub if c.hit)
                a = hh / len(sub)
                rt.add_row(h, str(len(sub)), str(hh), f"{a:.1%}", bar(a), "")
            g.add_row("", Text(""))
            g.add_row("ACTIVITY (regime head)",
                      Text(f"{rh}/{len(regime)} = {ra:.1%}", style="bold cyan"))
            g.add_row("vs 50/50 baseline",
                      Text(f"0.0% baseline   ({ra - 0.5:+.1%})",
                           style="green" if ra > 0.5 else "red"))
            t = Group(t, Text(""), rt)

        return Panel(Group(t, g), title="LIVE SCORECARD (this log)",
                     subtitle="[dim]measured, not asserted[/dim]",
                     border_style="yellow")

    def log_panel(self) -> Panel:
        rows = [c for c in self.score.calls if c.resolved][-8:][::-1]
        if not rows:
            return Panel(Text("nothing resolved yet", style="dim"),
                         title="RECENT CALLS", border_style="yellow")
        t = Table(box=None, pad_edge=False)
        t.add_column("made", style="dim", width=9)
        t.add_column("hor", width=5)
        t.add_column("call", width=9)
        t.add_column("conf", justify="right", width=6)
        t.add_column("expected", justify="right", width=10)
        t.add_column("actual", justify="right", width=10)
        t.add_column("", width=4)
        regime_shown = False
        for c in rows:
            if c.kind == "regime":
                regime_shown = True
                name = REGIME_CLASSES[c.predicted]
                shown = Text(f"{name[:3].lower()} {name}",
                             style="cyan" if name == "ACTIVE" else "dim")
                actual_txt = Text(REGIME_CLASSES[c.realized],
                                  style=("cyan" if c.realized else "dim"))
                # The realized figure is the forward mean |15m return|, not a
                # return sign - labelling it "% gain" would misread it.
                realized = Text(f"{c.realized_pct:.3f}%/bar", style="dim")
            else:
                trend = DIRECTION_CLASSES[c.predicted]
                shown = Text(f"{TREND_MARK[trend]} {trend}",
                             style=TREND_STYLE[trend])
                actual_txt = Text(DIRECTION_CLASSES[c.realized],
                                  style=TREND_STYLE[DIRECTION_CLASSES[c.realized]])
                realized = Text(f"{c.realized_pct:+.3f}%",
                                style="green" if c.realized_pct >= 0 else "red")
            t.add_row(
                short_ts(int(to_epoch_ms([pd.Timestamp(c.made_at)])[0])),
                c.horizon, shown,
                f"{c.confidence:.1%}",
                f"{c.expected_pct:+.3f}%" if c.kind == "direction" else "-",
                realized,
                Text("hit" if c.hit else "miss",
                     style="green" if c.hit else "red"),
            )
        note = ("actual class = threshold of the realized move using the frozen "
                "train thresholds")
        if regime_shown:
            note += ("; activity rows show the realized mean |15m move| per "
                     "candle and are scored against that call's own trailing "
                     "24h reference")
        return Panel(Group(t, Text(note, style="dim")),
                     title="RECENT CALLS", border_style="yellow")

    def model_panel(self) -> Panel:
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", justify="right")
        g.add_column()
        for k, v in (self.model_rows or [("status", self.status)]):
            g.add_row(k, v)
        return Panel(g, title="MODEL", border_style="green")

    def footer(self) -> Panel:
        parts = [
            "[bold]r[/bold]efresh", "[bold]i[/bold]nterval",
            "[bold]c[/bold]onfidence", "[bold]m[/bold]odel",
            "[bold]l[/bold]og", "[bold]x[/bold] clear", "[bold]q[/bold]uit",
        ]
        line = Text.from_markup("   ".join(parts))
        clock = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        status = self.status
        style = "white"
        if self.error:
            status, style = self.error[:70], "red"
        elif status != "live":
            style = "yellow"

        g = Table.grid(expand=True)
        g.add_column(justify="left")
        g.add_column(justify="right")
        g.add_row(line, Text(f"{clock}  ·  {status}", style=style))
        g.add_row(
            Text("Research viewer only - no orders, no fees, no slippage "
                 "modelled. A directional call is not a profitability claim "
                 "and this is not investment advice.", style="dim italic"),
            Text(f"up {time.time() - self.started:.0f}s", style="dim"),
        )
        return Panel(g, border_style="grey35")

    # -- assembly -----------------------------------------------------------

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="price", size=5),
            Layout(name="prediction", size=11),
            Layout(name="score", size=11 if self.score.resolved() else 7),
            Layout(name="log", size=13 if self.show_log else 0),
            Layout(name="footer", size=5),
        )
        layout["header"].update(self.header())
        layout["price"].update(self.price_panel())
        layout["prediction"].update(self.prediction_panel())
        layout["score"].update(self.score_panel())
        if self.show_log:
            layout["log"].update(self.log_panel())
        layout["footer"].update(self.footer())
        return layout

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        self._ensure_model()
        self.refresh()

        with Keys() as keys, Live(self.render(), console=self.console,
                                  screen=True, auto_refresh=False) as live:
            last_draw = 0.0
            while True:
                now = time.time()

                if (now - self.last_fetch) >= self.intervals[self.interval_idx]:
                    live.update(self.render(), refresh=True)
                    self.refresh()

                key = keys.get()
                if key:
                    if key in ("q", "Q", "\x03", "\x1b"):
                        break
                    elif key in ("r", "R"):
                        live.update(self.render(), refresh=True)
                        self.refresh()
                    elif key in ("i", "I"):
                        self.interval_idx = (self.interval_idx + 1) % len(self.intervals)
                        self.last_fetch = time.time()
                    elif key in ("c", "C"):
                        i = self.conf_choices.index(self.min_conf) \
                            if self.min_conf in self.conf_choices else 0
                        self.min_conf = self.conf_choices[
                            (i + 1) % len(self.conf_choices)]
                    elif key in ("m", "M"):
                        self.model_idx = (self.model_idx + 1) % len(self.models)
                        try:
                            self._ensure_model()
                            self.refresh()
                        except Exception as exc:            # noqa: BLE001
                            self.error = f"{type(exc).__name__}: {exc}"
                    elif key in ("l", "L"):
                        self.show_log = not self.show_log
                    elif key in ("x", "X"):
                        self.score.clear()
                        self.last_issue_ts = None
                        self._save_log()

                if now - last_draw > 0.2:
                    live.update(self.render(), refresh=True)
                    last_draw = now
                time.sleep(0.05)

        self._save_log()
        self.console.print(
            f"\n[session ended] {len(self.score.calls)} calls logged, "
            f"{len(self.score.resolved())} resolved.\n"
            f"Track record saved to {self.log_path.name}\n"
            f"Research viewer only. No orders were placed, no fees or slippage "
            f"are modelled, and a directional call is not a profitability "
            f"claim.\n")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Interactive terminal dashboard for the trained trend model.")
    p.add_argument("--interval", type=int, default=30,
                   help="seconds between refreshes (snapped to the nearest "
                        "available step; 'i' cycles at runtime)")
    p.add_argument("--model", default=None,
                   help="path to a .keras file (default: the GRU in models/)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="below this, a directional call is displayed as NEUTRAL")
    p.add_argument("--once", action="store_true",
                   help="fetch, predict, print one frame and exit (no UI)")
    args = p.parse_args()

    dash = Dashboard(args)
    if args.once:
        dash._ensure_model()
        dash.refresh()
        dash.console.print(dash.header())
        dash.console.print(dash.price_panel())
        dash.console.print(dash.prediction_panel())
        dash.console.print(dash.score_panel())
        return 0

    try:
        dash.run()
    except KeyboardInterrupt:
        dash._save_log()
        dash.console.print("\n[dim]interrupted - track record saved.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
