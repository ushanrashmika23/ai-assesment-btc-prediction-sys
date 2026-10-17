"""
Start the API, hit every endpoint, and say whether the payload is sane.

    python src/_smoke_api.py                 # the shipped ensemble
    python src/_smoke_api.py --model models/gru_s42_best.keras
    python src/_smoke_api.py --serve-web

Why this exists as a script rather than a curl transcript in a log: the two
paths through `Dashboard._ensure_model` are different code. The single-model
path loads one .keras; the ensemble path builds `Ensemble([...])` from the
members the bundle names. A smoke test that only ever ran the first path does
not exercise the default one, and the default is what ships. So the script
takes the model as an argument and the assertions below are written against
whichever path was asked for.

It is deliberately dependency-free (stdlib only, no `requests`) so it can run
on a machine that has nothing but the project's own requirements installed.

Exit code is 0 only if every endpoint answered AND every assertion held.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = Path(__file__).resolve().parent / "server.py"

# TensorFlow import alone is ~10s on this machine, then the model loads and the
# first REST top-up fetches candles. 240s is generous without being a hang.
BOOT_TIMEOUT = 240.0
REQUEST_TIMEOUT = 120.0

failures: list[str] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def get(url: str, timeout: float = REQUEST_TIMEOUT):
    """GET returning (status, body_bytes). HTTP errors are returned, not raised."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<38} {detail}")
    if not ok:
        failures.append(label)


def wait_ready(base: str, proc: subprocess.Popen) -> dict | None:
    """Poll /api/health until it answers. None if the server died or timed out."""
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  server exited early with code {proc.returncode}")
            return None
        try:
            status, body = get(f"{base}/api/health", timeout=5.0)
            if status == 200:
                return json.loads(body)
        except Exception:                                     # noqa: BLE001
            pass                       # not up yet: connection refused, no route
        time.sleep(1.0)
    print(f"  server did not answer /api/health within {BOOT_TIMEOUT:.0f}s")
    return None


def read_sse(url: str, want: int, timeout: float = 90.0) -> tuple[list[str], str]:
    """Read `want` `event: snapshot` frames. Returns (bodies, note)."""
    bodies: list[str] = []
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            deadline = time.time() + timeout
            while len(bodies) < want and time.time() < deadline:
                line = r.readline()
                if not line:                              # stream closed
                    return bodies, "stream closed"
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if text.startswith("data: "):
                    bodies.append(text[6:])
    except Exception as exc:                                  # noqa: BLE001
        return bodies, f"{type(exc).__name__}: {exc}"
    return bodies, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", default=None,
                    help="a single .keras to pass through to the server; "
                         "omit to test the shipped ensemble")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--serve-web", action="store_true",
                    help="also exercise the single-URL static mount")
    ap.add_argument("--no-record", action="store_true", default=True,
                    help="do not write a dashboard call log (on by default)")
    args = ap.parse_args()

    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"

    cmd = [sys.executable, str(SERVER), "--port", str(port)]
    if args.model:
        cmd += ["--model", args.model]
    if args.serve_web:
        cmd += ["--serve-web"]
    if args.no_record:
        cmd += ["--no-record"]

    expect_ensemble = args.model is None
    print(f"$ {' '.join(cmd)}")
    print(f"  expecting: {'ENSEMBLE (inference_bundle.json)' if expect_ensemble else args.model}")
    print()

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    try:
        health = wait_ready(base, proc)
        if health is None:
            out = proc.stdout.read() if proc.stdout else ""
            print("---- server output ----")
            print(out)
            return 1

        print("endpoints:")
        # -- the thing under test ------------------------------------------
        want_model = "inference_bundle.json" if expect_ensemble else Path(args.model).name
        check("health.model", health.get("model") == want_model,
              f"{health.get('model')!r}")

        status, body = get(f"{base}/api/snapshot")
        check("GET /api/snapshot", status == 200, f"{status}  {len(body) / 1024:.0f} KB")
        snap = json.loads(body)

        reg = snap.get("regime") or {}
        regime_hz = reg.get("horizons") or []
        check("snapshot.regime.horizons", len(regime_hz) == 4,
              f"{regime_hz}  tau={reg.get('tau')}")

        pred = snap.get("prediction") or {}
        horizons = pred.get("horizons") or []
        check("snapshot.prediction.horizons", len(horizons) == 6,
              f"{len(horizons)} horizons")

        # The direction and regression heads cover every horizon. The regime
        # head does not - it exists only at the horizons in regime.horizons, so
        # requiring `activity` at 45m/75m would be asserting something the
        # model was never trained to answer.
        ALWAYS = ("direction", "direction_probs", "expected_return_pct",
                  "projected_close", "band")
        missing = [(h.get("label"), k) for h in horizons
                   for k in ALWAYS if h.get(k) is None]
        check("every horizon has direction + regression",
              not missing, f"missing {missing}" if missing else
              f"{len(horizons)}/{len(horizons)}")
        act_missing = [h.get("label") for h in horizons
                       if h.get("label") in regime_hz and h.get("activity") is None]
        act_extra = [h.get("label") for h in horizons
                     if h.get("label") not in regime_hz and h.get("activity") is not None]
        check("activity exactly where the regime head exists",
              not act_missing and not act_extra,
              f"missing {act_missing} unexpected {act_extra}"
              if (act_missing or act_extra) else f"at {regime_hz}")

        for h in horizons:
            if h.get("direction") not in ("UP", "NEUTRAL", "DOWN"):
                failures.append(f"horizon {h.get('label')} direction={h.get('direction')!r}")
            if h.get("activity") not in (None, "QUIET", "ACTIVE"):
                failures.append(f"horizon {h.get('label')} activity={h.get('activity')!r}")

        curve = pred.get("curve") or []
        naive = pred.get("naive_curve") or []
        want = len(horizons) + 1                 # anchor + one per horizon
        check("snapshot.prediction.curve", len(curve) == want, f"{len(curve)} points")
        check("snapshot.prediction.naive_curve", len(naive) == want, f"{len(naive)} points")

        # The whole point of shipping both lines: the flat baseline must be
        # present so the model's projection is never drawn alone.
        if curve and naive:
            check("curve starts at the anchor, not the first horizon",
                  curve[0].get("ts") == naive[0].get("ts"))
            closes = [p.get("close") for p in curve]
            check("curve closes are finite and positive",
                  all(isinstance(c, (int, float)) and c > 0 for c in closes))

        # The recorded-projection block. It is read from SQLite on every
        # refresh, so the shape has to hold whether the store is full, empty
        # or absent - and a whitespace point (a time with no value) must stay
        # distinguishable from a point whose value happens to be missing.
        hist = snap.get("prediction_history")
        check("snapshot.prediction_history present", isinstance(hist, dict),
              "missing" if hist is None else f"{type(hist).__name__}")
        if isinstance(hist, dict):
            pts = hist.get("points")
            check("history.points is a list", isinstance(pts, list),
                  f"{len(pts)} point(s)" if isinstance(pts, list) else "not a list")
            if isinstance(pts, list):
                bad = [p for p in pts if not isinstance(p.get("time"), int)]
                times = [p["time"] for p in pts if isinstance(p.get("time"), int)]
                check("every history point has an integer time", not bad,
                      f"{len(bad)} bad" if bad else f"{len(times)} point(s)")
                check("history times ascend", times == sorted(times))
                gaps = [p for p in pts if "value" not in p]
                withval = [p for p in pts if "value" in p]
                check("history values are finite where present",
                      all(isinstance(p["value"], (int, float))
                          and p["value"] == p["value"] for p in withval),
                      f"{len(withval)} priced, {len(gaps)} whitespace")
                # NOT "is one of snapshot.candles": a projection names a candle
                # 1-6 ahead of its anchor, so the newest points are always past
                # the end of the price history - that is what a projection is.
                # 15m candle boundaries are the invariant that does hold.
                offgrid = [p["time"] for p in withval if p["time"] % 900]
                check("history points sit on 15m boundaries", not offgrid,
                      f"off-grid: {offgrid[:3]}" if offgrid else f"{len(withval)} point(s)")
                last_candle = max((c["time"] for c in snap.get("candles") or []), default=0)
                ahead = [p["time"] for p in withval if p["time"] > last_candle]
                check("history may run past the last candle",
                      all(t > last_candle for t in ahead),
                      f"{len(ahead)} projected point(s) beyond the price history")
            check("history reports its gap rule", isinstance(hist.get("gap_candles"), int),
                  f"gap_candles={hist.get('gap_candles')}")

        met = snap.get("metrics") or {}
        check("snapshot.metrics", bool(met.get("key")),
              f"key={met.get('key')!r} split={met.get('split')!r} "
              f"n={met.get('n_samples')}")
        if expect_ensemble:
            check("metrics keyed to the ensemble", met.get("key") == "ensemble_x3",
                  f"{met.get('key')!r}")

        check("snapshot.candles", len(snap.get("candles") or []) > 100,
              f"{len(snap.get('candles') or [])} candles")

        summary = {r[0]: r[1] for r in (snap.get("model_summary") or [])}
        if expect_ensemble:
            check("model_summary says ENSEMBLE",
                  "ENSEMBLE" in str(summary.get("architecture", "")),
                  f"{summary.get('architecture')!r} "
                  f"members={summary.get('members')!r}")

        # -- the rest -------------------------------------------------------
        for path, key in [("/api/candles?limit=20", "candles"),
                          ("/api/prediction", "prediction"),
                          ("/api/track-record", None),
                          ("/api/metrics", None)]:
            status, body = get(f"{base}{path}")
            ok = status == 200
            detail = f"{status}"
            if ok and key:
                try:
                    n = len(json.loads(body).get(key) or [])
                    detail += f"  {key}={n}"
                except Exception:                             # noqa: BLE001
                    ok = False
            elif ok:
                detail += f"  {len(body)} bytes"
            check(f"GET {path}", ok, detail)

        check("GET /api/candles?limit=5 -> 422", get(f"{base}/api/candles?limit=5")[0] == 422,
              "(below the endpoint's ge=10 floor)")

        # -- SSE ------------------------------------------------------------
        print("\nstream:")
        frames, note = read_sse(f"{base}/api/stream", want=2)
        check("GET /api/stream", len(frames) == 2, f"{len(frames)} frames, {note}")
        if frames:
            f0 = json.loads(frames[0])
            check("stream frame parses", isinstance(f0, dict),
                  f"{len(frames[0]) / 1024:.0f} KB")
            check("stream frame has ok + prediction",
                  "ok" in f0 and "prediction" in f0)

        # -- optional static mount ------------------------------------------
        if args.serve_web:
            print("\nstatic mount:")
            import re
            status, body = get(f"{base}/")
            html = body.decode("utf-8", "replace")
            check("GET / serves index.html",
                  status == 200 and 'id="root"' in html,
                  f"{status}  {len(body)} bytes")
            for asset in re.findall(r'(?:src|href)="(/assets/[^"]+)"', html):
                st, _ = get(f"{base}{asset}")
                check(f"GET {asset}", st == 200, str(st))
            check("/api/health not shadowed by the mount",
                  get(f"{base}/api/health")[0] == 200)
            check("GET /docs", get(f"{base}/docs")[0] == 200)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"\nserver stopped (code {proc.returncode})")

    print()
    if failures:
        print(f"FAILED - {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
