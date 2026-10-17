"""
T06, T07, T08 -- the HTTP API.

The app is built by ``server.build_app(engine, web_dist)``; every route is a
closure over the ``engine`` it is handed, so the whole HTTP surface can be
exercised by handing it a different engine.

What is mocked, and what that costs
-----------------------------------
``_StubEngine`` replaces **LiveEngine only**, and only the parts of it that
touch the outside world: the Binance REST fetch, the TensorFlow inference, and
the background recorder thread. It does not replace, reimplement or shortcut
anything else -- the FastAPI app, the route bodies, the ``Query`` validation
that produces the 422s in T07, the SSE framing in T08, and the JSON
serialisation are all the project's own code.

The payload it serves is not invented either: it is ``web/_snapshot.json``, the
real capture the project commits for its own SSR checks. So the *content*
assertions are made against a payload the live server actually produced.

The engine is a stub rather than the real thing because starting ``LiveEngine``
costs a TensorFlow import, a network round-trip and a recorder thread -- none
of which belongs in a deterministic test. The live engine is still covered, by
one separate test at the bottom of this file that skips with an explicit reason
when the market feed or the model bundle is unavailable.

Read-only guarantee
-------------------
``LiveEngine(record=False)`` sets ``PredictionStore(create=False)`` (SQLite
``mode=ro``) and makes ``start_recorder()`` a no-op, so no test in this file can
write to ``models/``.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import _support
import httpx
import pytest

pytest.importorskip("fastapi", reason="T06-T08 need the API's dependencies")

from fastapi.testclient import TestClient  # noqa: E402

from server import CACHE_SECONDS, build_app  # noqa: E402

# The five read endpoints T06 names, plus health (the endpoint that reports
# whether the thing behind them is alive at all).
SNAPSHOT_ENDPOINTS = ["/api/snapshot", "/api/candles", "/api/prediction",
                      "/api/track-record", "/api/metrics"]

CANDLE_MS = 15 * 60 * 1000


# ---------------------------------------------------------------------------
# the stub engine
# ---------------------------------------------------------------------------

class _StubEngine:
    """
    LiveEngine's interface, backed by a captured payload instead of the market.

    Only the four attributes ``build_app`` reads (``dash``, ``last_error``,
    ``model_ready``, ``started``), the recorder pair the lifespan calls, and
    ``snapshot()`` are implemented -- deliberately not a mock library, so an
    attribute the server starts needing shows up as an AttributeError rather
    than as a silently-mocked empty value.
    """

    def __init__(self, payload: dict, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.started = time.time()
        self.last_error = None
        self.model_ready = True
        self.dash = SimpleNamespace(
            model_name=str(payload.get("model", "inference_bundle.json")),
            status="live",
            error=None,
            last_fetch=payload.get("generated_at"),
        )
        self.snapshot_calls: list[float | None] = []
        self.recorder_starts = 0
        self.recorder_stops = 0

    # the lifespan calls exactly these two
    def start_recorder(self) -> None:
        self.recorder_starts += 1

    def stop_recorder(self) -> None:
        self.recorder_stops += 1

    # routes call this; /api/snapshot and /api/stream pass max_age, the rest
    # call it bare
    def snapshot(self, max_age: float | None = None) -> dict:
        self.snapshot_calls.append(max_age)
        if self.error is not None:
            raise self.error
        return self.payload


@pytest.fixture(scope="module")
def api(captured_snapshot):
    """A TestClient over the real app with the stub engine behind it."""
    engine = _StubEngine(captured_snapshot)
    app = build_app(engine, None)          # web_dist=None: API only, no static mount
    with TestClient(app) as client:        # `with` runs the real lifespan
        client.engine = engine             # handy for the wiring assertions
        yield client


# ---------------------------------------------------------------------------
# SSE frame parsing
# ---------------------------------------------------------------------------

def _frames(lines, cap: int = 8):
    """
    Reassemble SSE frames from ``iter_lines()`` output.

    Implements the one part of the spec this project uses: ``event:`` and
    ``data:`` fields, a ``:`` comment line, and a blank line ending the frame.
    Written out rather than pulled from a library so the test asserts against
    the wire format itself, which is what a browser's EventSource parses.
    """
    event: str | None = None
    data: list[str] = []
    comment: str | None = None
    count = 0

    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if event is not None or data or comment is not None:
                yield SimpleNamespace(event=event, data="\n".join(data),
                                      comment=comment)
                count += 1
                if count >= cap:
                    return
            event, data, comment = None, [], None
            continue
        if line.startswith(":"):
            comment = line[1:].strip()
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def _read_frames(response, n: int = 1, cap: int = 8):
    """The first ``n`` frames, then stop reading -- the stream never ends."""
    out = []
    for frame in _frames(response.iter_lines(), cap=cap):
        out.append(frame)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# T06 -- the read endpoints
# ---------------------------------------------------------------------------

def test_T06_every_read_endpoint_answers_200_with_json(api):
    """
    T06 -- snapshot, candles, prediction, track-record and metrics all return
    HTTP 200 and a JSON body that is not empty.
    """
    for path in SNAPSHOT_ENDPOINTS:
        response = api.get(path)
        assert response.status_code == 200, (
            f"T06 FAILED: GET {path} returned {response.status_code}, "
            f"expected 200.\n  body: {response.text[:300]}"
        )
        assert response.headers["content-type"].startswith("application/json"), (
            f"GET {path} returned content-type "
            f"{response.headers['content-type']!r}"
        )
        body = response.json()
        assert body, f"GET {path} returned an empty body"

        # Round-trip through the serialiser: a body FastAPI could not encode
        # cannot have got this far, but a NaN or a numpy scalar that slipped
        # through would raise here rather than in the browser.
        json.dumps(body, allow_nan=False)


def test_T06_snapshot_carries_the_documented_sections(api, captured_snapshot):
    """
    T06 -- /api/snapshot is the payload the dashboard renders: the anchor
    candle, the 240 candles before it, and six horizon projections.
    """
    body = api.get("/api/snapshot").json()

    missing = set(captured_snapshot) - set(body)
    assert not missing, (
        f"the snapshot lost section(s) the captured payload has: {sorted(missing)}"
    )

    anchor = body["anchor"]
    assert anchor["candles_in_window"] >= 64, (
        f"the anchor reports only {anchor['candles_in_window']} candles in the "
        "window; the model needs at least the 64-candle lookback"
    )
    assert anchor["input_shape"] == [1, 64, 49], (
        f"anchor.input_shape is {anchor['input_shape']}, expected [1, 64, 49] "
        "(one sample, 64 candles, 49 features)"
    )
    assert anchor["timestamp_ms"] % CANDLE_MS == 0, (
        f"anchor.timestamp_ms {anchor['timestamp_ms']} is not on a 15m boundary"
    )

    candles = body["candles"]
    assert len(candles) == 240, (
        f"the snapshot carries {len(candles)} candles, expected the documented 240"
    )
    assert candles[-1]["time"] * 1000 == anchor["timestamp_ms"], (
        "the last candle in the snapshot is not the anchor candle: "
        f"{candles[-1]['time'] * 1000} vs {anchor['timestamp_ms']}"
    )


def test_T06_snapshot_horizons_are_well_formed(api):
    """
    T06 -- each of the six projections is internally consistent: probabilities
    that sum to one, a confidence equal to the probability of the class it
    chose, and a projected close that follows from the expected return.
    """
    body = api.get("/api/snapshot").json()
    horizons = body["prediction"]["horizons"]
    assert len(horizons) == 6, f"expected 6 horizons, got {len(horizons)}"

    close = body["anchor"]["close"]
    for h in horizons:
        where = f"horizon {h['label']}"

        probs = h["direction_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, (
            f"{where}: direction_probs sum to {total!r}, not 1.0 - they are "
            "softmax outputs and must be a distribution"
        )
        assert h["direction"] in probs, (
            f"{where}: direction {h['direction']!r} is not one of the classes "
            f"{sorted(probs)}"
        )
        assert abs(h["direction_confidence"] - probs[h["direction"]]) < 1e-9, (
            f"{where}: direction_confidence {h['direction_confidence']!r} is "
            f"not the probability of the chosen class "
            f"({probs[h['direction']]!r})"
        )

        expected = h["expected_return_pct"]
        implied = close * (1.0 + expected / 100.0)
        assert abs(implied - h["projected_close"]) < 0.01, (
            f"{where}: projected_close {h['projected_close']!r} does not follow "
            f"from anchor close {close} and expected_return_pct {expected!r} "
            f"(implied {implied!r})"
        )

        assert h["candles"] * CANDLE_MS == h["minutes"] * 60 * 1000, (
            f"{where}: {h['candles']} candles is not {h['minutes']} minutes"
        )

    steps = [h["candles"] for h in horizons]
    assert steps == sorted(steps) and len(set(steps)) == 6, (
        f"the horizons are not six distinct increasing steps: {steps}"
    )


def test_T06_the_snapshot_is_honest_about_the_regression_head(api):
    """
    T06 -- the caveat the project ships with the numbers must still be there.

    ``web/_snapshot.json`` records that the regression head is worse than
    predicting no change at every horizon. The API must keep saying so: if the
    caveat field ever goes missing or goes empty, the dashboard would present a
    known-bad number with nothing attached to it.
    """
    body = api.get("/api/snapshot").json()
    prediction = body["prediction"]

    assert prediction.get("caveat"), (
        "prediction.caveat is empty - the dashboard would render projected "
        "returns with no warning that the regression head underperforms the "
        "no-change baseline"
    )
    assert "WORSE" in prediction["caveat"].upper(), (
        f"the caveat no longer names the regression head's failure: "
        f"{prediction['caveat']!r}"
    )
    assert prediction.get("curve_note"), (
        "prediction.curve_note is empty - the projection chart has no caption"
    )
    assert prediction.get("curve"), "prediction.curve is empty"
    assert prediction.get("band_note"), (
        "prediction.band_note is empty - the shaded band would be drawn with "
        "no statement of what it is"
    )

    # The baseline the caveat names has to be on the chart, or the caption is
    # pointing at a line nobody can see.
    naive = prediction.get("naive_curve")
    assert naive, "the projection lost its naive_curve baseline"
    naive_points = [p for p in naive if p.get("kind") == "naive"]
    assert len(naive_points) == len(prediction["horizons"]), (
        f"the naive baseline covers {len(naive_points)} of "
        f"{len(prediction['horizons'])} horizons"
    )
    anchor_close = body["anchor"]["close"]
    for point in naive_points:
        assert point["close"] == anchor_close, (
            "the naive baseline is not the 'no change' line: it projects "
            f"{point['close']} from an anchor of {anchor_close}"
        )


def test_T06_candles_endpoint_returns_ohlcv(api):
    """
    T06 -- /api/candles returns a usable OHLCV series: times strictly ascending
    on the 15m grid, and every candle internally consistent (high >= both ends,
    low <= both ends, volume non-negative).
    """
    body = api.get("/api/candles").json()

    assert body["symbol"] == "BTCUSDT" and body["interval"] == "15m", (
        f"the endpoint is reporting {body.get('symbol')}/{body.get('interval')}"
    )

    candles = body["candles"]
    assert len(candles) == 240, f"expected the default 240 candles, got {len(candles)}"

    times = [c["time"] for c in candles]
    assert all(isinstance(t, int) for t in times), "candle times are not integers"
    assert times == sorted(set(times)), (
        "candle times are not strictly ascending and unique"
    )
    steps = {b - a for a, b in zip(times, times[1:])}
    assert steps == {900}, (
        f"candles are not on a contiguous 15m grid - step sizes seen: "
        f"{sorted(steps)}"
    )

    for c in candles:
        assert c["high"] >= max(c["open"], c["close"]), f"high < open/close: {c}"
        assert c["low"] <= min(c["open"], c["close"]), f"low > open/close: {c}"
        assert c["volume"] >= 0, f"negative volume: {c}"
        assert c["low"] <= c["high"], f"low above high: {c}"


def test_T06_candles_endpoint_honours_the_limit(api):
    """
    T06 -- the ``limit`` query parameter really slices, and never returns more
    than the snapshot holds.
    """
    for limit in (10, 50, 240):
        candles = api.get("/api/candles", params={"limit": limit}).json()["candles"]
        assert len(candles) == limit, (
            f"limit={limit} returned {len(candles)} candle(s)"
        )

    # 1000 is the documented maximum, but the snapshot only ever holds 240, so
    # the endpoint must return what exists rather than padding or 500ing.
    candles = api.get("/api/candles", params={"limit": 1000}).json()["candles"]
    assert len(candles) == 240, (
        f"limit=1000 returned {len(candles)} candles; the snapshot holds 240, "
        "so the route should return all of them"
    )

    # The slice is the TAIL: the most recent candles, ending on the anchor.
    full = api.get("/api/candles", params={"limit": 240}).json()["candles"]
    tail = api.get("/api/candles", params={"limit": 30}).json()["candles"]
    assert tail == full[-30:], "the limit is not slicing the most recent candles"


def test_T06_prediction_endpoint_is_a_subset_of_the_snapshot(api):
    """
    T06 -- /api/prediction returns anchor/prediction/regime/model and nothing
    else, and each of the four is exactly what the snapshot holds.
    """
    snapshot = api.get("/api/snapshot").json()
    body = api.get("/api/prediction").json()

    assert set(body) == {"anchor", "prediction", "regime", "model"}, (
        f"/api/prediction returned keys {sorted(body)}"
    )
    for key in ("anchor", "prediction", "regime", "model"):
        assert body[key] == snapshot[key], (
            f"/api/prediction.{key} differs from the snapshot's - the two "
            "endpoints are serving different views of the same payload"
        )


def test_T06_track_record_endpoint_reports_resolved_calls(api):
    """
    T06 -- /api/track-record reports the recorded calls and the per-kind hit
    rates, and the arithmetic is self-consistent.
    """
    body = api.get("/api/track-record").json()

    for key in ("n_calls", "n_resolved", "recent", "direction", "regime"):
        assert key in body, f"track_record is missing {key!r}"

    recent = body["recent"]
    assert recent, "the track record has no recent calls to chart"
    assert len(recent) <= 600, (
        f"{len(recent)} calls returned; the endpoint documents at most 600"
    )
    assert body["n_resolved"] <= body["n_calls"], (
        f"n_resolved {body['n_resolved']} exceeds n_calls {body['n_calls']}"
    )

    for entry in recent:
        for field in ("made_at", "horizon", "kind", "target_ts", "predicted",
                      "confidence", "price_at_call"):
            assert field in entry, (
                f"a recorded call is missing {field!r}: {sorted(entry)}"
            )
        assert 0.0 <= entry["confidence"] <= 1.0, (
            f"confidence {entry['confidence']} is not a probability"
        )
        assert entry["kind"] in ("direction", "regime"), (
            f"unknown call kind {entry['kind']!r}"
        )
        if entry.get("resolved"):
            assert entry.get("realized") is not None and entry.get("realized_pct") is not None, (
                f"a call is marked resolved but carries no outcome: {entry}"
            )

    for kind in ("direction", "regime"):
        block = body[kind]
        assert block["n"] >= 0 and block["hits"] >= 0
        assert block["hits"] <= block["n"], (
            f"{kind}: {block['hits']} hits out of {block['n']} calls"
        )
        if block["n"]:
            assert abs(block["accuracy"] - block["hits"] / block["n"]) < 1e-9, (
                f"{kind}: accuracy {block['accuracy']} is not hits/n "
                f"({block['hits']}/{block['n']})"
            )
        assert "by_horizon" in block


def test_T06_metrics_endpoint_reports_the_test_split(api):
    """
    T06 -- /api/metrics reports the held-out scores, and the regime head's
    claim is the one the project's acceptance target rests on: it beats its
    own baseline.
    """
    body = api.get("/api/metrics").json()

    assert body["split"] == "test", (
        f"metrics are reported for the {body['split']!r} split; anything other "
        "than the held-out test split is not a result"
    )
    assert body["n_samples"] == 46686, (
        f"{body['n_samples']} samples - the shipped test split holds 46,686"
    )

    direction = body["direction"]
    assert 0.0 <= direction["accuracy"] <= 1.0
    assert "baseline_accuracy" in direction, (
        "the direction head reports an accuracy with no baseline to read it "
        "against"
    )

    regime = body["regime"]
    assert regime["accuracy"] > regime["baseline_accuracy"], (
        f"the regime head scores {regime['accuracy']:.4f} against a "
        f"{regime['baseline_accuracy']:.4f} baseline - it no longer beats it, "
        "which is the one accuracy claim this project makes"
    )
    assert regime["lift"] > 0, f"regime lift is {regime['lift']}"

    per_horizon = regime["per_horizon"]
    assert per_horizon, "the regime block reports no per-horizon breakdown"
    for label, block in per_horizon.items():
        assert abs(block["lift"] - (block["accuracy"] - block["baseline"])) < 1e-9, (
            f"regime {label}: lift {block['lift']} is not accuracy - baseline "
            f"({block['accuracy']} - {block['baseline']})"
        )


def test_T06_health_endpoint_reports_the_engine_state(api):
    """
    T06 -- /api/health is what a supervisor polls, so it has to answer without
    touching the feed and name the bundle in use.
    """
    body = api.get("/api/health").json()

    assert body["ok"] is True, f"health reports not-ok: {body}"
    assert body["model_ready"] is True
    assert body["bundle"] == "inference_bundle.json", (
        f"health names bundle {body['bundle']!r}; the shipped inference bundle "
        "is inference_bundle.json"
    )
    assert isinstance(body["uptime_seconds"], (int, float))
    assert body["uptime_seconds"] >= 0
    assert "error" in body and "last_fetch" in body


def test_T06_the_routes_are_wired_to_the_engine(api):
    """
    T06 -- the routes really call the engine they were built with, and they
    pass the cache hint through. Without this the endpoint tests above could
    all be passing against a payload nobody asked the engine for.
    """
    before = len(api.engine.snapshot_calls)
    api.get("/api/snapshot", params={"max_age": 3})
    api.get("/api/track-record")

    calls = api.engine.snapshot_calls[before:]
    assert calls, "the routes returned a payload without calling engine.snapshot()"
    assert 3 in calls, (
        f"max_age=3 never reached the engine - it saw {calls}. The snapshot "
        "cache hint is being dropped."
    )
    assert None in calls, (
        f"the bare-snapshot routes passed {calls}; /api/track-record and "
        "friends call engine.snapshot() with no cache hint"
    )


def test_T06_the_lifespan_starts_and_stops_the_recorder(captured_snapshot):
    """
    T06 -- the recorder thread is tied to the app's lifetime, not to import or
    to the first request: ``build_app``'s lifespan must both start it and stop
    it, or a long-running server leaks a thread on every reload.

    Deliberately its own short-lived app rather than the module-wide ``api``
    client: the module client's teardown has not run yet while any test is
    executing, so only a nested ``with`` can observe the pairing.
    """
    engine = _StubEngine(captured_snapshot)

    with TestClient(build_app(engine, None)) as client:
        assert client.get("/api/health").status_code == 200
        assert engine.recorder_starts == 1, (
            f"the lifespan called start_recorder() {engine.recorder_starts} "
            "time(s) on entry; expected exactly once"
        )
        assert engine.recorder_stops == 0, (
            "the recorder was stopped while the app was still serving"
        )

    assert engine.recorder_stops == engine.recorder_starts == 1, (
        f"start_recorder() was called {engine.recorder_starts} time(s) but "
        f"stop_recorder() {engine.recorder_stops} - the lifespan's teardown "
        "does not pair with its startup"
    )


def test_T06_a_dead_engine_degrades_the_api_without_crashing_it(captured_snapshot):
    """
    T06 -- the failure path. With a feed that raises, the JSON endpoints must
    surface an error rather than hang or return a half-built payload; the
    stream must say so in a frame (T08 checks the frame; here, that the app
    survives it).
    """
    engine = _StubEngine(captured_snapshot, error=RuntimeError("feed unavailable"))
    with TestClient(build_app(engine, None), raise_server_exceptions=False) as client:
        for path in SNAPSHOT_ENDPOINTS:
            response = client.get(path)
            assert response.status_code >= 500, (
                f"GET {path} returned {response.status_code} while the engine "
                "was raising; an unreachable feed is not a 200"
            )
            assert response.status_code < 600, (
                f"GET {path} returned {response.status_code}, which is not a "
                "server error code"
            )


# ---------------------------------------------------------------------------
# T07 -- limit validation
# ---------------------------------------------------------------------------

def test_T07_limit_five_is_rejected_with_422(api):
    """
    T07 -- GET /api/candles?limit=5 must be refused with HTTP 422, because the
    documented minimum is 10.
    """
    response = api.get("/api/candles", params={"limit": 5})

    assert response.status_code == 422, (
        f"T07 FAILED: /api/candles?limit=5 returned {response.status_code} "
        f"(expected 422).\n  body: {response.text[:400]}"
    )

    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail, (
        f"the 422 body has no validation detail: {response.text[:300]}"
    )
    entry = detail[0]
    assert entry["loc"][-1] == "limit", (
        f"the 422 blames {entry['loc']} rather than the limit parameter"
    )
    assert entry["type"] == "greater_than_equal", (
        f"the 422 is {entry['type']!r}, expected greater_than_equal"
    )
    assert entry["ctx"]["ge"] == 10, (
        f"the 422 cites a minimum of {entry['ctx'].get('ge')}, expected 10"
    )

    # Distinguish a rejection from a silent clamp: the response must be an
    # error, not a 200 carrying five candles.
    assert "candles" not in response.text, (
        "the request was refused but the body still contains candle data"
    )


@pytest.mark.parametrize("limit,expected", [
    (0, 422), (1, 422), (5, 422), (9, 422),          # below the minimum
    (10, 200), (11, 200), (240, 200), (1000, 200),   # inside the range
    (1001, 422), (10_000, 422),                      # above the maximum
])
def test_T07_the_limit_range_is_ten_to_a_thousand(api, limit, expected):
    """
    T07 -- the boundary either side of the documented range. 10 and 1000 are
    the inclusive edges and must be accepted; one outside must not.
    """
    response = api.get("/api/candles", params={"limit": limit})
    assert response.status_code == expected, (
        f"limit={limit} returned {response.status_code}, expected {expected}.\n"
        f"  body: {response.text[:300]}"
    )
    if expected == 200:
        assert response.json()["candles"], (
            f"limit={limit} was accepted but returned no candles"
        )


@pytest.mark.parametrize("limit", ["abc", "5.5", "", "null", "1e3"])
def test_T07_a_non_integer_limit_is_rejected(api, limit):
    """
    T07 -- the parameter is typed ``int``; anything that is not one is a 422,
    not a coercion, and never a 500.
    """
    response = api.get("/api/candles", params={"limit": limit})
    assert response.status_code == 422, (
        f"limit={limit!r} returned {response.status_code}, expected 422.\n"
        f"  body: {response.text[:300]}"
    )


def test_T07_the_published_schema_declares_the_same_range(api):
    """
    T07 -- read the OpenAPI schema the app publishes and check the 422s above
    come from the declared contract, not from a coincidence. This is the
    assertion that would catch someone widening ``ge=10`` to ``ge=1`` while a
    test still happened to pass.
    """
    schema = api.get("/openapi.json").json()
    params = schema["paths"]["/api/candles"]["get"]["parameters"]

    limit = next((p for p in params if p["name"] == "limit"), None)
    assert limit is not None, "/api/candles no longer documents a limit parameter"
    assert limit["schema"]["minimum"] == 10, (
        f"the documented minimum is {limit['schema'].get('minimum')}, expected 10"
    )
    assert limit["schema"]["maximum"] == 1000, (
        f"the documented maximum is {limit['schema'].get('maximum')}, expected 1000"
    )
    assert limit["schema"]["default"] == 240, (
        f"the documented default is {limit['schema'].get('default')}, expected 240"
    )


def test_T07_the_snapshot_cache_hint_is_validated_too(api):
    """
    T07 -- the same guard on the other query parameter, so the validation is a
    property of the API rather than of one lucky route.
    """
    assert api.get("/api/snapshot", params={"max_age": -1}).status_code == 422
    assert api.get("/api/snapshot", params={"max_age": 601}).status_code == 422
    assert api.get("/api/snapshot", params={"max_age": 0}).status_code == 200
    assert api.get("/api/snapshot",
                   params={"max_age": 600}).status_code == 200
    assert api.get("/api/snapshot",
                   params={"max_age": CACHE_SECONDS}).status_code == 200


# ---------------------------------------------------------------------------
# T08 -- the event stream
#
# These four tests talk to a REAL uvicorn server on a loopback port, not to
# TestClient. That is not a preference: TestClient buffers each response body
# before returning it, so it can never return from /api/stream, which by design
# never ends. A real server is also the only thing that shows the bytes a
# browser's EventSource would actually parse.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sse_available():
    """Skip the whole T08 group, loudly, if no server can be started."""
    pytest.importorskip("uvicorn", reason="T08 needs a real HTTP server")
    return True


@contextlib.contextmanager
def _served(payload: dict, *, error: Exception | None = None):
    """Build the app over a stub engine and serve it on a free port."""
    engine = _StubEngine(payload, error=error)
    app = build_app(engine, None)
    try:
        with _support.serve_app(app) as base:
            yield base, engine
    except _support.ServerUnavailable as exc:
        pytest.skip(f"BLOCKED: no HTTP server available: {exc}")


def test_T08_the_stream_speaks_server_sent_events(sse_available,
                                                  captured_snapshot):
    """
    T08 -- the headers a browser's EventSource requires, and a first frame that
    declares itself a snapshot.
    """
    with _served(captured_snapshot) as (base, _engine):
        with httpx.Client(timeout=30.0) as client:
            with client.stream("GET", f"{base}/api/stream",
                               params={"max_age": 0}) as response:
                assert response.status_code == 200, (
                    f"the stream returned {response.status_code}"
                )
                content_type = response.headers["content-type"]
                assert content_type.startswith("text/event-stream"), (
                    f"content-type is {content_type!r}; EventSource requires "
                    "text/event-stream"
                )
                assert response.headers.get("cache-control") == "no-cache", (
                    "an SSE response that can be cached will replay stale frames"
                )

                frames = _read_frames(response, n=1)

    assert len(frames) == 1, "the stream produced no frame within the first event"
    frame = frames[0]
    assert frame.event == "snapshot", (
        f"the first frame is {frame.event!r}, expected 'snapshot'"
    )
    payload = json.loads(frame.data)
    assert payload["symbol"] == "BTCUSDT"
    assert "anchor" in payload and "prediction" in payload


def test_T08_the_streamed_frame_matches_the_rest_snapshot(sse_available,
                                                          captured_snapshot):
    """
    T08 -- what the client actually depends on: the ``data:`` line is valid
    JSON and describes the same payload ``/api/snapshot`` serves, so a client
    that only reads the stream sees what a polling client sees.
    """
    with _served(captured_snapshot) as (base, _engine):
        with httpx.Client(timeout=30.0) as client:
            rest = client.get(f"{base}/api/snapshot").json()
            with client.stream("GET", f"{base}/api/stream",
                               params={"max_age": 0}) as response:
                frames = _read_frames(response, n=1)

    payload = json.loads(frames[0].data)
    for key in ("anchor", "prediction", "candles", "regime", "model"):
        assert key in payload, (
            f"the streamed snapshot has no {key!r} - a client that only reads "
            "the stream would render an empty panel"
        )
    assert payload == rest, (
        "the streamed frame and the REST snapshot disagree. They are built "
        f"from the same engine.snapshot() call; differing keys: "
        f"{sorted(set(rest) ^ set(payload)) or 'same keys, different values'}"
    )

    # The stream sends compact JSON on ONE line; a raw newline inside the data
    # would be read by EventSource as the end of the frame.
    assert "\n" not in frames[0].data, (
        "the data line contains a raw newline, which would split the frame"
    )


def test_T08_an_unchanged_snapshot_is_sent_as_a_keepalive(sse_available,
                                                          captured_snapshot):
    """
    T08 -- duplicate suppression. The engine is polled again on the next tick;
    when it returns an identical payload the stream must send a comment
    heartbeat rather than repeating the whole snapshot -- that is what keeps a
    proxied connection open without re-sending 240 candles every 12 seconds.
    """
    with _served(captured_snapshot) as (base, _engine):
        with httpx.Client(timeout=30.0) as client:
            with client.stream("GET", f"{base}/api/stream",
                               params={"max_age": 0}) as response:
                frames = _read_frames(response, n=2, cap=4)

    assert len(frames) == 2, (
        f"expected two frames (one snapshot, one heartbeat), got {len(frames)}"
    )
    assert frames[1].event is None and frames[1].comment, (
        f"the second frame is {frames[1]!r}; an unchanged snapshot must arrive "
        "as a ':' comment heartbeat"
    )
    assert frames[1].comment == "keepalive", (
        f"the heartbeat comment is {frames[1].comment!r}, expected 'keepalive'"
    )
    assert not frames[1].data, "the heartbeat frame must not carry data"


def test_T08_a_failing_engine_produces_an_error_frame(sse_available,
                                                      captured_snapshot):
    """
    T08 -- the error path. A stream cannot return a 500 once it has started, so
    the failure has to be delivered in-band, as a frame the client can parse
    and act on.
    """
    with _served(captured_snapshot,
                 error=RuntimeError("feed unavailable")) as (base, _engine):
        with httpx.Client(timeout=30.0) as client:
            with client.stream("GET", f"{base}/api/stream",
                               params={"max_age": 0}) as response:
                assert response.status_code == 200, (
                    "the stream refused the connection instead of reporting "
                    "the failure in a frame"
                )
                frames = _read_frames(response, n=1, cap=4)

    assert frames, "a failing engine produced no frame at all"
    frame = frames[0]
    assert frame.event == "error", (
        f"the first frame is {frame.event!r}, expected 'error'"
    )
    assert frame.data, "the error frame carries no data"
    payload = json.loads(frame.data)
    assert "error" in payload, (
        f"the error frame's JSON has no 'error' key: {payload}"
    )
    assert "feed unavailable" in payload["error"], (
        f"the error frame lost the cause: {payload['error']!r}"
    )
    assert "RuntimeError" in payload["error"], (
        "the error frame no longer names the exception type"
    )


def test_T08_the_refresh_interval_has_a_floor(sse_available, captured_snapshot):
    """
    T08 -- ``max_age=0`` must not become a busy loop: the generator sleeps at
    least two seconds between frames whatever the client asks for. Two frames
    at max_age=0 therefore cannot arrive back-to-back.
    """
    with _served(captured_snapshot) as (base, _engine):
        with httpx.Client(timeout=30.0) as client:
            with client.stream("GET", f"{base}/api/stream",
                               params={"max_age": 0}) as response:
                started = time.monotonic()
                frames = _read_frames(response, n=2, cap=4)
                elapsed = time.monotonic() - started

    assert len(frames) == 2, f"only {len(frames)} frame(s) arrived"
    assert elapsed >= 1.5, (
        f"two frames arrived {elapsed:.2f}s apart at max_age=0; the stream is "
        "not throttling and would hammer the feed"
    )
    assert elapsed < 60.0, (
        f"two frames took {elapsed:.1f}s; the refresh loop appears to be stuck"
    )


# ---------------------------------------------------------------------------
# T06 -- the live engine, end to end
# ---------------------------------------------------------------------------

def test_T06_the_real_live_engine_serves_the_same_contract(captured_snapshot):
    """
    T06 -- the same endpoints, with the real ``LiveEngine`` behind them instead
    of the stub: a live Binance fetch and a real TensorFlow inference.

    This is the one test in the suite that needs the network. It is written to
    SKIP -- loudly, with the reason -- when the feed or the model bundle is
    unavailable, because a missing network is an environment fact, not a defect
    in the code under test. It is never reported as a pass when it did not run.

    ``record=False`` is what keeps it safe to run at any time: no recorder
    thread starts, the store is opened read-only, and the JSON track record is
    not written.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "inference_bundle.json").exists():
        pytest.skip(
            f"BLOCKED: {models_dir / 'inference_bundle.json'} is missing, so "
            "there is no model to serve. Run `python src/train.py` first."
        )

    try:
        from server import LiveEngine
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"BLOCKED: server.LiveEngine could not be imported: "
                    f"{type(exc).__name__}: {exc}")

    # -- phase 1: the engine itself. Any failure here is the environment.
    try:
        engine = LiveEngine(record=False)
        live = engine.snapshot(0)
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(
            f"BLOCKED: the live engine could not produce a snapshot "
            f"({type(exc).__name__}: {exc}). This test needs the Binance REST "
            "feed and the TensorFlow runtime; the stub-backed tests above cover "
            "the same endpoints without either."
        )

    assert live.get("ok") is True, (
        f"the live engine produced a snapshot with ok={live.get('ok')!r}: "
        f"{str(live.get('error'))[:300]}"
    )
    assert not engine.record, "record=False must not enable the recorder"
    assert engine.store is not None and not engine.store._writable, (
        "record=False must open the prediction store read-only"
    )

    # -- phase 2: the HTTP surface. A failure from here on is a real failure.
    with TestClient(build_app(engine, None)) as client:
        for path in SNAPSHOT_ENDPOINTS:
            response = client.get(path)
            assert response.status_code == 200, (
                f"T06 FAILED against the LIVE engine: GET {path} returned "
                f"{response.status_code}.\n  body: {response.text[:300]}"
            )
            assert response.json(), f"GET {path} returned an empty body live"

        # The live payload must satisfy the same structural contract the
        # captured one did.
        body = client.get("/api/snapshot").json()
        assert len(body["prediction"]["horizons"]) == 6
        assert body["anchor"]["input_shape"] == [1, 64, 49]
        candles = client.get("/api/candles").json()["candles"]
        assert len(candles) == 240
        times = [c["time"] for c in candles]
        assert times == sorted(set(times)), "live candles are out of order"

    engine.stop_recorder()   # no-op when record=False, but pair the call anyway
