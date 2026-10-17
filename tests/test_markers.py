"""
T13 -- chart markers built from out-of-order calls.

The markers are built by ``web/src/markers.js::buildMarkers()``, which the
React chart calls with ``snapshot.track_record.recent``. That array is not
sorted and not unique -- by construction: a window is written as its six
direction horizons (targets +1..+6 candles) and then its four activity
horizons (targets +1, +2, +4, +6), so the target timestamps climb and then drop
back to +1, and four of the ten land on a candle an earlier call already
claimed. Lightweight Charts asserts that marker times are *strictly* ascending,
so both properties used to crash the chart.

The documented behaviour, and what T13 asks about, is: **10 calls over 6
distinct candles produce 6 merged markers**, sorted, with the calls sharing a
candle merged into one marker whose text is a tally (``2/2``, ``1/2``).

Two layers, deliberately:

* ``tests/node/marker_probe.mjs`` drives the real JS function with a
  hand-computed one-window fixture and reports JSON. It must not be
  reimplemented in Python -- a Python copy would only prove the copy agrees
  with itself.
* this file asserts the same headline numbers *again*, from the raw marker
  arrays the probe emits, so the result is not merely the JS test vouching for
  itself. It also runs the project's own ``web/markers-check.mjs``, which works
  from the real captured payload instead of a synthetic window.

Node is an external dependency. It is not mocked or stubbed: if it is missing
these tests report BLOCKED, with the reason.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import _support

PROBE = _support.NODE_DIR / "marker_probe.mjs"
PROJECT_CHECK = _support.WEB_DIR / "markers-check.mjs"

# The canonical tally, copied from the fixture table in marker_probe.mjs.
EXPECTED_TALLY = ["0/2", "1/2", "●", "2/2", "●", "2/2"]


@pytest.fixture(scope="module")
def probe():
    """
    Run tests/node/marker_probe.mjs once and parse its JSON report.

    Module-scoped: it is the same run for every assertion below.
    """
    if shutil.which("node") is None:
        pytest.skip(
            "BLOCKED: node is not on PATH, so the dashboard's marker builder "
            "cannot be exercised at all."
        )
    if not PROBE.exists():
        pytest.skip(f"BLOCKED: {PROBE} is missing")

    proc = _support.run_node(PROBE, cwd=_support.PROJECT_ROOT, timeout=180.0)

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"the marker probe did not print JSON ({exc}).\n"
            f"exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[:2000]}\n"
            f"--- stderr ---\n{proc.stderr[:2000]}"
        ) from exc

    evidence = _support.write_evidence(
        "T13_marker_probe_report",
        json.dumps(report, indent=2) + "\n\n--- stderr ---\n" + proc.stderr,
    )
    print(f"T13 evidence: {evidence}")
    return report, proc


# ---------------------------------------------------------------------------
# T13
# ---------------------------------------------------------------------------

def test_T13_ten_out_of_order_calls_produce_six_merged_markers(probe):
    """
    T13 -- the headline expectation: 10 calls across 6 distinct candle targets
    produce 6 markers, not 10 and not 4.
    """
    report, _proc = probe

    fixture = report["fixture"]
    assert fixture["calls"] == 10, (
        f"the fixture issues {fixture['calls']} calls, not the 10 T13 is about"
    )
    assert fixture["distinct_candles"] == 6, (
        f"the fixture's calls land on {fixture['distinct_candles']} candles, "
        "not 6"
    )

    canonical = report["canonical"]
    assert len(canonical) == 6, (
        f"T13 FAILED: 10 calls over 6 distinct candles produced "
        f"{len(canonical)} marker(s), expected 6 merged markers.\n"
        f"  markers: {json.dumps(canonical)}"
    )

    # ...and 6 is the *merged* count: the calls are all still represented.
    counted = sum(int(m["text"].split("/")[1]) if "/" in m["text"] else 1
                  for m in canonical)
    assert counted == 10, (
        f"the 6 markers account for {counted} of the 10 calls - merging "
        "dropped some rather than combining them"
    )


def test_T13_the_markers_are_strictly_ascending(probe):
    """
    T13 -- sorted, and strictly so: the chart's ``setMarkers()`` rejects equal
    times as well as out-of-order ones, so "sorted" is not enough.

    This is asserted against the raw times the probe returned, so it holds even
    if the probe's own assertion is ever loosened.
    """
    report, _proc = probe
    times = [m["time"] for m in report["canonical"]]

    assert all(isinstance(t, (int, float)) for t in times), (
        f"non-numeric marker times: {times}"
    )
    assert times == sorted(times), (
        f"T13 FAILED: the markers are not in ascending time order.\n"
        f"  times: {times}\n"
        "  a marker array that is not ascending is rejected by the chart"
    )
    assert len(set(times)) == len(times), (
        f"T13 FAILED: two markers share a timestamp, which the chart's "
        f"strictly-ascending assertion also rejects.\n  times: {times}"
    )

    # ...and every marker sits on a candle that exists in the series.
    candle_ms = report["fixture"]["candle_ms"]
    anchor = report["fixture"]["anchor_ms"]
    axis = {(anchor + k * candle_ms) // 1000 for k in range(1, 7)}
    off_axis = [t for t in times if t not in axis]
    assert not off_axis, (
        f"marker(s) at {off_axis} are not on the candle series the chart is "
        f"drawing; the axis holds {sorted(axis)}"
    )


def test_T13_each_marker_reports_its_tally(probe):
    """
    T13 -- merged correctly, part 1: the text. A candle two calls landed on
    says how many of them were right (``2/2``, ``1/2``, ``0/2``); a candle one
    call landed on keeps the single-call glyph it always had.
    """
    report, _proc = probe
    got = [m["text"] for m in report["canonical"]]

    assert got == EXPECTED_TALLY, (
        f"T13 FAILED: the marker texts are {got}, expected {EXPECTED_TALLY}.\n"
        "  For one window the ten calls land on candles +1..+6 and the tallies "
        "are 0/2, 1/2, single, 2/2, single, 2/2."
    )

    singles = [m for m in report["canonical"] if "/" not in m["text"]]
    assert len(singles) == 2, (
        f"{len(singles)} single-call marker(s); candles +3 and +5 are named by "
        "the direction head only"
    )
    for marker in singles:
        assert marker["text"] == "●", (
            f"a lone direction call is drawn as {marker['text']!r}, expected "
            "the ● it has always been"
        )


def test_T13_position_and_colour_follow_the_tally(probe):
    """
    T13 -- merged correctly, part 2: what the marker looks like.

    The rules the module documents: a tie goes below the bar and a call the
    model got wrong goes above it; the colour is green only when every call at
    that candle landed, red when none did, amber in between. Asserted against
    the palette the probe read out of ``format.js``, so this does not hard-code
    hex values that are free to change.
    """
    report, _proc = probe
    palette = report["palette"]
    by_text = {m["text"]: m for m in report["canonical"]}

    assert by_text["2/2"]["color"] == palette["up"], (
        f"2/2 is {by_text['2/2']['color']}, expected the up colour "
        f"{palette['up']}"
    )
    assert by_text["1/2"]["color"] == palette["neutral"], (
        f"1/2 is {by_text['1/2']['color']}, expected the neutral colour "
        f"{palette['neutral']} - two values disagreeing is the interesting case"
    )
    assert by_text["0/2"]["color"] == palette["down"], (
        f"0/2 is {by_text['0/2']['color']}, expected the down colour "
        f"{palette['down']}"
    )

    assert by_text["1/2"]["position"] == "belowBar", (
        "1/2 is drawn above the bar; a tie (hits*2 >= n) must go below it"
    )
    assert by_text["0/2"]["position"] == "aboveBar", (
        "0/2 is drawn below the bar; a call the model got wrong belongs above"
    )
    assert by_text["2/2"]["position"] == "belowBar"

    shape_bad = [m for m in report["canonical"] if m["shape"] != "circle"]
    assert not shape_bad, f"unexpected marker shape(s): {shape_bad}"


def test_T13_the_probe_ran_every_check_and_they_all_passed(probe):
    """
    T13 -- the probe's own verdict: order-independence across 12 input
    orderings, no call silently dropped, hostile input skipped rather than
    thrown on, and the degenerate payloads returning an empty array.
    """
    report, proc = probe

    assert report["n_checks"] >= 13, (
        f"the probe ran only {report['n_checks']} checks; expected the full set"
    )
    failed = [c for c in report["checks"] if not c["ok"]]
    assert not failed, (
        "T13 FAILED: the marker probe reported "
        f"{len(failed)} failing check(s):\n  "
        + "\n  ".join(f"{c['name']}: {c['detail']}" for c in failed)
    )
    assert report["failures"] == 0
    assert proc.returncode == 0, (
        f"the probe exited {proc.returncode} with no failing check recorded"
    )


def test_T13_the_probes_fixture_really_is_the_pathological_one(probe):
    """
    T13 -- guard against a vacuous pass. If the fixture were sorted and unique,
    every assertion above would still hold while testing nothing: the bug being
    guarded against needs an out-of-order array with duplicate targets.
    """
    report, _proc = probe
    fixture_check = next(
        (c for c in report["checks"]
         if c["name"].startswith("the fixture really is out of order")), None)
    assert fixture_check is not None, (
        "the probe no longer verifies that its own fixture is pathological"
    )
    assert fixture_check["ok"], (
        f"the fixture is no longer out of order with collisions: "
        f"{fixture_check['detail']}"
    )
    assert "backwards step" in (fixture_check["detail"] or ""), (
        f"the fixture check reports {fixture_check['detail']!r}; expected it to "
        "report the backwards step it reproduced"
    )


def test_T13_the_projects_own_marker_check_still_passes():
    """
    T13 -- ``web/markers-check.mjs``, the project's own guard, run as a
    subprocess against the real captured payload.

    It covers ground this file does not: the marker output is derived from
    ``web/_snapshot.json`` rather than a synthetic window, so a change in what
    the server records is exercised too. Run rather than reimplemented -- the
    project's check is the thing that has to keep passing.
    """
    if shutil.which("node") is None:
        pytest.skip("BLOCKED: node is not on PATH")
    if not PROJECT_CHECK.exists():
        pytest.skip(f"BLOCKED: {PROJECT_CHECK} is missing")

    proc = _support.run_node(PROJECT_CHECK, cwd=_support.WEB_DIR, timeout=180.0)
    output = (proc.stdout or "") + (proc.stderr or "")

    evidence = _support.write_evidence(
        "T13_project_markers_check",
        f"$ node web/markers-check.mjs   (cwd={_support.WEB_DIR})\n"
        f"exit code: {proc.returncode}\ncaptured: {_support.now_iso()}\n"
        + "=" * 78 + "\n" + output,
    )
    print(f"T13 evidence: {evidence}")
    print(output.strip())

    assert proc.returncode == 0, (
        f"T13 FAILED: web/markers-check.mjs exited {proc.returncode}.\n"
        + output[-3000:]
    )
    assert "FAIL" not in output, (
        "the project's marker check printed a failure:\n" + output[-3000:]
    )
    assert "all marker checks passed" in output, (
        "the project's marker check did not report success; output was:\n"
        + output[-2000:]
    )


def test_T13_the_captured_payload_still_exercises_the_merge():
    """
    T13 -- the captured payload is the fixture the project's own check runs on.
    If it ever stopped containing duplicate target timestamps, the merge branch
    would stop being exercised there and the check would pass without testing
    anything.
    """
    snapshot = _support.read_json(_support.CAPTURED_SNAPSHOT)
    recent = (snapshot.get("track_record") or {}).get("recent") or []
    if not recent:
        pytest.skip("the captured payload holds no recorded calls")

    targets = [c.get("target_ts") for c in recent if c.get("target_ts") is not None]
    distinct = set(targets)

    assert len(targets) > len(distinct), (
        f"the {len(targets)} captured calls all name distinct candles "
        f"({len(distinct)}), so nothing in the payload exercises the merge.\n"
        "  This is not a failure of the code - it means the fixture has gone "
        "stale and web/markers-check.mjs is no longer testing the case it was "
        "written for."
    )

    # The regression is that the raw array is unsorted; confirm it still is, or
    # say plainly that the ordering half of the bug can no longer be shown from
    # this fixture (the project's own check rebuilds issue order for that
    # reason, so this assertion is informational).
    ordered = all(b >= a for a, b in zip(targets, targets[1:]))
    print(
        f"T13 fixture: {len(recent)} captured calls over {len(distinct)} "
        f"candles ({len(targets) - len(distinct)} collision(s)); "
        f"raw order is {'sorted' if ordered else 'out of order'}"
    )


def test_T13_the_browser_bundle_contains_the_marker_builder(paths):
    """
    T13 -- the module under test is the one the dashboard actually ships. A
    passing Node check against ``web/src/markers.js`` would mean nothing if the
    built bundle under ``web/dist`` were stale, because the browser loads that.
    """
    if not paths.web_dist.exists():
        pytest.skip(
            f"{paths.web_dist} is missing - run `npm run build` in web/ to "
            "produce the bundle the browser loads"
        )

    assets = sorted((paths.web_dist / "assets").glob("*.js"))
    if not assets:
        pytest.skip("web/dist has no JS assets to inspect")

    # The bundle is minified, so the marker's own literals are what identify
    # it: the two positions and the 'size' key only ever appear together here.
    markers_src = (_support.WEB_DIR / "src" / "markers.js").read_text(encoding="utf-8")
    assert "belowBar" in markers_src and "aboveBar" in markers_src

    bundled = any("belowBar" in blob.read_text(encoding="utf-8", errors="replace")
                  and "aboveBar" in blob.read_text(encoding="utf-8", errors="replace")
                  for blob in assets)
    assert bundled, (
        f"none of {[a.name for a in assets]} contains the marker builder's "
        "output, so web/dist predates web/src/markers.js.\n"
        "  Run `npm run build` in web/ - otherwise the browser is running a "
        "different marker builder from the one T13 verified."
    )
