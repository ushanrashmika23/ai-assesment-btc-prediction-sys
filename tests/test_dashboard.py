"""
T11, T12 -- the two dashboards.

This project has two front ends, and the two cases are about different ones:

* **T11 -- the terminal dashboard** (``src/dashboard.py``). Every panel is a
  method returning a rich renderable, so all of them can be rendered in-process
  with no terminal, no network and no model. Rendered twice: once filled from
  the real captured payload, once with nothing loaded at all. The first proves
  the panels draw; the second proves they degrade instead of raising, which is
  exactly the state the live loop is in for the first second after start-up and
  after every feed failure.

* **T12 -- the React app** (``web/``). Its simple/advanced split is asserted by
  the project's own ``web/check.mjs`` (run here as a subprocess, not
  reimplemented), and then the built bundle is served over loopback and driven
  in a real headless browser: the default view is screenshotted, the toggle is
  clicked, and the advanced view is screenshotted.

Evidence
--------
Every render is exported under ``tests/screenshots/``: panel text and HTML for
T11, PNGs and page text for T12. The browser is the only optional dependency in
this file -- when it is unavailable the T12 browser test reports as skipped
with the reason, while the assertions that do not need a browser still run, and
still fail if the app ever opens on the advanced view.

What is substituted, and what is not
------------------------------------
Only two things are replaced, both outside production code:

* the rich ``Console`` the module prints through, swapped for a recording one
  at a fixed width so the exported evidence is stable and comparable. Same
  class, same module, different sink -- no project function is patched.
* ``dashboard.MODELS_DIR``, pointed at a temp directory in the one test that
  proves a dry run does not append to the track record. That test writes to
  ``tmp_path``, and the real ``models/`` directory is asserted unchanged.

The data is not mocked: it is the real captured ``/api/snapshot`` payload the
project already uses as its own SSR fixture.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import _support

pytest.importorskip("rich", reason="T11 renders rich panels")

# ``dashboard`` reads the module-level CONSOLE at construction time, so the
# recording console has to be installed before any Dashboard is built.
import dashboard as dashboard_module  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.panel import Panel  # noqa: E402

CONSOLE_WIDTH = 150

# The read-only panels: no arguments, no terminal, no network.
PANELS = ("header", "price_panel", "prediction_panel", "score_panel",
          "log_panel", "model_panel", "footer")

# The one panel with no title: it is the key-hint / status bar.
UNTITLED = {"footer"}

HORIZON_LABELS = ["15m", "30m", "45m", "60m", "75m", "90m"]


def _require_dashboard_inputs() -> None:
    """
    T11 needs the frozen metadata and a model file to construct a Dashboard --
    the constructor raises SystemExit without either. Checked here so a project
    that has not been trained yet reports BLOCKED instead of taking the pytest
    process down with it.
    """
    if not dashboard_module.FINAL_METADATA.exists():
        pytest.skip(
            f"BLOCKED: {dashboard_module.FINAL_METADATA} is missing, so the "
            "dashboard has no frozen thresholds/classes to read. Run "
            "`python src/pipeline.py` first."
        )
    if not dashboard_module.MODELS_DIR.exists():
        pytest.skip(f"BLOCKED: {dashboard_module.MODELS_DIR} does not exist")
    if not (dashboard_module.INFERENCE_BUNDLE.exists()
            or any(dashboard_module.MODELS_DIR.glob("*.keras"))):
        pytest.skip(
            f"BLOCKED: no model in {dashboard_module.MODELS_DIR}, and the "
            "dashboard refuses to start without one."
        )


@pytest.fixture(scope="module", autouse=True)
def recording_console():
    """
    Replace the module console with a recording one at a fixed width.

    This is the only substitution T11 makes, and it replaces nothing but the
    output sink: it is rich's own Console class, the same one the module uses,
    with ``record=True`` so each panel can be exported as evidence. A fixed
    width is what makes that evidence comparable between runs.
    """
    _require_dashboard_inputs()
    original = dashboard_module.CONSOLE
    console = Console(record=True, width=CONSOLE_WIDTH, height=60,
                      legacy_windows=False, force_terminal=False,
                      color_system=None)
    dashboard_module.CONSOLE = console
    yield console
    dashboard_module.CONSOLE = original


def _args(**overrides):
    """The arguments ``src/dashboard.py``'s CLI would supply."""
    base = dict(model=None, interval=30, min_confidence=0.0, once=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def _new_dashboard(**overrides):
    """A Dashboard with no data loaded -- the state the live loop starts in."""
    return dashboard_module.Dashboard(_args(**overrides))


def _draw(console: Console, panel) -> str:
    """Render one panel and return just its text, leaving the buffer clean."""
    console.print(panel)
    return console.export_text(clear=True)


def _load_snapshot_into(dash, snapshot: dict) -> None:
    """
    Populate a Dashboard from the captured ``/api/snapshot`` payload, using the
    shapes the live refresh path assigns: ``candles`` is a DataFrame with a
    ``close`` column, ``probs`` is (6, 3) in DIRECTION_CLASSES order, and so on.
    """
    candles = pd.DataFrame(snapshot["candles"])
    candles["timestamp"] = pd.to_datetime(candles["time"], unit="s", utc=True)
    dash.candles = candles

    anchor = snapshot["anchor"]
    dash.live = SimpleNamespace(
        last_close=float(anchor["close"]),
        last_timestamp=pd.Timestamp(int(anchor["timestamp_ms"]), unit="ms",
                                    tz="UTC"),
        n_candles=int(anchor["candles_in_window"]),
    )
    dash.model_name = str(snapshot.get("model") or "inference_bundle.json")
    dash.status = str(snapshot.get("status") or "live")
    dash.error = None
    dash.last_fetch = time.time()

    horizons = snapshot["prediction"]["horizons"]
    labels = [h["label"] for h in horizons]
    assert labels == HORIZON_LABELS, (
        f"the captured prediction block is in {labels} order, expected "
        f"{HORIZON_LABELS}; the panel indexes probs by HORIZON_ORDER"
    )

    dash.returns = np.array([h["expected_return_pct"] for h in horizons],
                            dtype=float)
    # DIRECTION_CLASSES is {index: name}, so the array has to be built in index
    # order -- argmax() is read back against that same table.
    dash.probs = np.array(
        [[h["direction_probs"][dashboard_module.DIRECTION_CLASSES[i]]
          for i in sorted(dashboard_module.DIRECTION_CLASSES)]
         for h in horizons], dtype=float)

    # The regime head answers a subset of the horizons (15m/30m/60m/90m), listed
    # in the snapshot's regime block; the panel maps horizon -> row through the
    # Dashboard's own regime_horizons, so rows must be built in that order.
    by_label = {h["label"]: h for h in horizons}
    regime_horizons = list(dash.regime_horizons)
    assert regime_horizons, (
        "the frozen metadata declares no regime horizons, so the activity half "
        "of the prediction panel cannot be exercised"
    )
    dash.regime_probs = np.array(
        [[by_label[label]["activity_probs"][
            dashboard_module.REGIME_CLASSES[i]]
          for i in sorted(dashboard_module.REGIME_CLASSES)]
         for label in regime_horizons], dtype=float)
    dash.ref_absmove = float(snapshot["regime"]["reference_absmove"])

    # The store's projection of a call (what /api/track-record returns) carries
    # neither the model name nor the candle distance nor the regression figure,
    # so those three come from where the payload does state them: the name from
    # the snapshot, the distance from the horizon's own definition, and the
    # expected return from the prediction block for the same horizon. Nothing is
    # invented; every field is read out of the payload.
    expected_by_label = {h["label"]: float(h["expected_return_pct"])
                         for h in horizons}
    candles_by_label = dashboard_module.CANDLES_BY_LABEL

    dash.score = dashboard_module.Scoreboard()
    calls = []
    for entry in snapshot["track_record"]["recent"]:
        if not entry.get("resolved"):
            continue
        label = entry.get("horizon")
        if label not in candles_by_label:
            continue
        calls.append(dashboard_module.Call(
            made_at=entry["made_at"], model=dash.model_name,
            horizon=label, candles=candles_by_label[label],
            target_ts=int(entry["target_ts"]), predicted=int(entry["predicted"]),
            confidence=float(entry["confidence"]),
            expected_pct=expected_by_label.get(label, 0.0),
            price_at_call=float(entry["price_at_call"]),
            realized_pct=entry.get("realized_pct"),
            realized=entry.get("realized"),
            kind=entry.get("kind") or "direction",
            ref_absmove=entry.get("ref_absmove"),
        ))
    dash.score.add(calls)
    assert calls, "the captured payload held no resolved calls to populate with"


# ---------------------------------------------------------------------------
# T11 -- the terminal dashboard
# ---------------------------------------------------------------------------

def test_T11_every_panel_renders_with_real_data(recording_console, paths,
                                                captured_snapshot):
    """
    T11 -- render all panels with real data. Expected: no exception, and every
    component renders something.
    """
    dash = _new_dashboard()
    _load_snapshot_into(dash, captured_snapshot)

    drawn: dict[str, str] = {}
    for name in PANELS:
        panel = getattr(dash, name)()
        assert panel is not None, f"Dashboard.{name}() returned None"
        drawn[name] = _draw(recording_console, panel)

    for name, text in drawn.items():
        assert text.strip(), (
            f"T11 FAILED: Dashboard.{name}() produced no output at all with "
            "real data loaded"
        )

    # It must be drawing the payload, not falling back to a placeholder.
    joined = "\n".join(drawn.values())
    for placeholder in ("waiting for the first candle fetch", "no candles yet",
                        "no prediction yet", "nothing resolved yet"):
        assert placeholder not in joined, (
            f"a panel fell back to its empty state ({placeholder!r}) even "
            "though the real payload was loaded"
        )

    assert dashboard_module.SYMBOL in drawn["header"], (
        "the header does not name the symbol it is showing"
    )
    for label in HORIZON_LABELS:
        assert label in drawn["prediction_panel"], (
            f"the prediction panel does not list the {label} horizon"
        )
    assert ("ACTIVE" in drawn["prediction_panel"]
            or "QUIET" in drawn["prediction_panel"]), (
        "the activity head's read is missing from the prediction panel"
    )

    text = _support.write_evidence(
        "T11_terminal_dashboard_real_data",
        f"captured: {_support.now_iso()}\n"
        f"console: rich Console(record=True, width={CONSOLE_WIDTH})\n"
        f"source: {_support.CAPTURED_SNAPSHOT}\n"
        + "=" * CONSOLE_WIDTH + "\n"
        + "\n".join(drawn.values()) + "\n",
    )
    print(f"T11 evidence: {text}")


def test_T11_the_whole_layout_renders_with_real_data(recording_console,
                                                     captured_snapshot):
    """
    T11 -- ``render()`` assembles the panels into the Layout the live loop
    draws. Every component has to survive being placed in it: a panel that
    renders fine on its own can still fail once rich is told to fit it into a
    fixed-height slot.
    """
    dash = _new_dashboard()
    _load_snapshot_into(dash, captured_snapshot)

    layout = dash.render()
    assert isinstance(layout, Layout)
    recording_console.print(layout)
    out = recording_console.export_text(clear=True)

    assert out.strip(), "render() produced no output"
    assert dashboard_module.SYMBOL in out, (
        "the assembled layout does not contain the header"
    )
    for label in HORIZON_LABELS:
        assert label in out, f"the assembled layout is missing the {label} row"

    html = recording_console.export_html(inline_styles=True, clear=False)
    html_path = _support.write_html_evidence(
        "T11_terminal_dashboard_real_data", html)
    png = _support.stamp_png("T11_terminal_dashboard_real_data")
    svg_ok = True
    try:
        png.with_suffix(".svg").write_text(
            recording_console.export_svg(title="T11 terminal dashboard"),
            encoding="utf-8")
    except Exception as exc:                                    # noqa: BLE001
        svg_ok = False
        print(f"T11: rich could not export SVG ({exc}); text and HTML kept")
    print(f"T11 evidence: {html_path} (SVG: {svg_ok}) | {png}")


def test_T11_the_panels_are_well_formed_renderables(recording_console,
                                                    captured_snapshot):
    """
    T11 -- the containers, not just the pixels: each panel is a rich ``Panel``,
    and ``render()`` places them in the slots it declares. Asserting the types
    is what stops a panel from silently returning a plain string that merely
    happens to print.
    """
    dash = _new_dashboard()
    _load_snapshot_into(dash, captured_snapshot)

    for name in PANELS:
        panel = getattr(dash, name)()
        assert isinstance(panel, Panel), (
            f"Dashboard.{name}() returned {type(panel).__name__}, expected a "
            "rich Panel"
        )
        if name not in UNTITLED:
            assert panel.title is not None, (
                f"Dashboard.{name}() has no title, so the panel is unlabelled "
                "on screen"
            )

    layout = dash.render()
    names = [child.name for child in layout.children]
    assert names == ["header", "price", "prediction", "score", "log",
                     "footer"], (
        f"the layout slots are {names}, expected header/price/prediction/"
        "score/log/footer"
    )


def test_T11_every_panel_degrades_with_no_data(recording_console, paths):
    """
    T11 -- the other half of the case: render all panels with *empty* data.
    Expected: no exception, and every component still renders its own
    placeholder -- this is the state the live loop is in for the first second
    after start-up and after any feed failure.
    """
    dash = _new_dashboard()
    assert dash.candles is None and dash.live is None, (
        "a freshly constructed Dashboard already holds candles"
    )
    assert dash.probs is None and dash.returns is None

    # "Empty" here means no *market data*: no candles, no prediction. The
    # scoreboard is deliberately NOT empty -- the constructor re-loads the
    # selected model's persisted track record so the tally accumulates across
    # restarts. That reload is worth asserting in its own right, and then the
    # scoreboard is cleared, because the genuinely cold start (a model that has
    # never resolved a call) is the state the placeholders exist for.
    log_path = dash.log_path
    if log_path.exists():
        on_disk = _support.read_json(log_path).get("calls", [])
        assert len(dash.score.calls) == len(on_disk), (
            f"the dashboard loaded {len(dash.score.calls)} of the {len(on_disk)}"
            f" calls in {log_path.name}; the track record is not being restored "
            "whole"
        )
        assert len(dash.score.resolved()) == sum(
            1 for c in on_disk if c.get("realized") is not None), (
            "the number of resolved calls does not match the track record"
        )
        print(f"T11: restored {len(on_disk)} calls from {log_path.name}")
    dash.score.clear()
    assert dash.score.resolved() == []

    drawn: dict[str, str] = {}
    for name in PANELS:
        try:
            panel = getattr(dash, name)()
        except Exception as exc:                                # noqa: BLE001
            raise AssertionError(
                f"T11 FAILED: Dashboard.{name}() raised "
                f"{type(exc).__name__}: {exc} with no data loaded.\n"
                "  The dashboard must render its empty state, not crash: this "
                "is the state it starts in."
            ) from exc
        assert panel is not None, f"Dashboard.{name}() returned None"
        drawn[name] = _draw(recording_console, panel)

    for name, text in drawn.items():
        assert text.strip(), (
            f"T11 FAILED: Dashboard.{name}() produced no output at all with no "
            "data loaded"
        )

    assert "waiting for the first candle fetch" in drawn["header"], (
        "the header does not say it is waiting for the first fetch"
    )
    assert dashboard_module.SYMBOL in drawn["header"], (
        "the empty header no longer names the symbol/interval it waits on"
    )
    assert "no candles yet" in drawn["price_panel"], (
        "the price panel does not say it has no candles"
    )
    assert "no prediction yet" in drawn["prediction_panel"], (
        "the prediction panel does not say it has no prediction"
    )
    assert "nothing resolved yet" in drawn["log_panel"], (
        "the recent-calls panel does not say nothing has resolved"
    )
    assert "calls issued" in drawn["score_panel"], (
        "the empty scorecard does not report how many calls were issued"
    )

    try:
        recording_console.print(dash.render())
        assert recording_console.export_text(clear=True).strip()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"T11 FAILED: render() raised {type(exc).__name__}: {exc} with no "
            "data loaded"
        ) from exc

    text = _support.write_evidence(
        "T11_terminal_dashboard_empty_data",
        f"captured: {_support.now_iso()}\n"
        "state: a Dashboard with no candles, no live input and no prediction, "
        "and an empty scoreboard (the cold-start state)\n"
        + "=" * CONSOLE_WIDTH + "\n"
        + "\n".join(drawn.values()) + "\n",
    )
    print(f"T11 evidence: {text}")


def test_T11_partial_and_error_states_render(recording_console,
                                             captured_snapshot):
    """
    T11 -- the states between "nothing" and "everything", which the live loop
    passes through on every refresh: a direction read without the regime head,
    a stale fetch, a feed error, and a one-candle frame. None of them may
    raise.
    """
    # 1. a payload with no regime head behind it (a build that predates it).
    partial = _new_dashboard()
    _load_snapshot_into(partial, captured_snapshot)
    partial.regime_probs = None
    partial.regime_horizons = []
    partial.ref_absmove = None
    for name in PANELS:
        assert _draw(recording_console, getattr(partial, name)()).strip()

    # 2. a stale fetch: the header still counts down to the next one.
    stale = _new_dashboard()
    _load_snapshot_into(stale, captured_snapshot)
    stale.last_fetch = time.time() - 10_000
    for name in PANELS:
        assert _draw(recording_console, getattr(stale, name)()).strip()

    # 3. a feed error: the footer turns red and shows the message.
    broken = _new_dashboard()
    broken.status = "error"
    broken.error = "RuntimeError: feed unavailable"
    for name in PANELS:
        _draw(recording_console, getattr(broken, name)())
    footer = _draw(recording_console, broken.footer())
    assert "feed unavailable" in footer, (
        "the footer does not surface the engine's error message"
    )

    # 4. a one-candle frame: len(candles) > 1 is what the header guards on.
    tiny = _new_dashboard()
    _load_snapshot_into(tiny, captured_snapshot)
    tiny.candles = tiny.candles.iloc[[-1]]
    assert len(tiny.candles) == 1
    assert _draw(recording_console, tiny.header()).strip()
    assert _draw(recording_console, tiny.price_panel()).strip()


def test_T11_a_dry_run_does_not_write_to_the_track_record(paths, tmp_path,
                                                          monkeypatch):
    """
    T11 -- ``once=True`` (``--once``) is a dry run and must not append to the
    track record. The suite constructs dashboards with ``once=True`` throughout,
    so this is the assertion that keeps the rest of the file from mutating
    ``models/``.

    Includes the positive control: the same call with recording ON does write,
    into a temp directory -- so the test cannot pass by ``_save_log`` simply
    being broken.
    """
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
              for p in paths.models.glob("dashboard_log_*.json")}

    dry = _new_dashboard(once=True)
    assert dry.record is False, (
        "Dashboard(args.once=True).record is not False - a dry run would write "
        "to the track record"
    )
    live = _new_dashboard(once=False)
    assert live.record is True, (
        "Dashboard(args.once=False).record is not True; the control below "
        "would not prove anything"
    )

    # Redirect the track record's directory for the rest of this test. This
    # patches the module global the log path is built from, and nothing else;
    # both dashboards are already constructed, so no model lookup is affected.
    monkeypatch.setattr(dashboard_module, "MODELS_DIR", tmp_path)

    dry._save_log()                       # the call the refresh loop makes
    assert not list(tmp_path.glob("*.json")), (
        "T11 FAILED: a --once (dry-run) dashboard wrote "
        f"{[p.name for p in tmp_path.glob('*.json')]} to the track record"
    )

    live._save_log()
    written = sorted(p.name for p in tmp_path.glob("dashboard_log_*.json"))
    assert written, (
        "the positive control failed: _save_log() wrote nothing even with "
        "recording on, so the dry-run assertion above proves nothing"
    )
    print(f"T11 control: a recording dashboard writes {written}")

    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in paths.models.glob("dashboard_log_*.json")}
    assert before == after, (
        "the real models/ track record changed during this test:\n"
        f"  before: {before}\n  after:  {after}"
    )


def test_T11_the_scoreboard_separates_the_two_heads(recording_console,
                                                    captured_snapshot):
    """
    T11 -- the score panel's own rule: the direction head and the regime head
    are scored against different baselines and must never be pooled. Pooling
    them would move the headline toward 0.50 without either head changing, so
    the panel splits them -- and this asserts it has both heads to split.
    """
    dash = _new_dashboard()
    _load_snapshot_into(dash, captured_snapshot)

    resolved = dash.score.resolved()
    assert resolved, "the scoreboard resolved nothing from the payload"
    kinds = {c.kind for c in resolved}
    assert kinds <= {"direction", "regime"}, f"unexpected call kind(s): {kinds}"
    assert kinds == {"direction", "regime"}, (
        "the captured payload no longer exercises both heads, so the panel's "
        "separation logic is not being drawn"
    )

    text = _draw(recording_console, dash.score_panel())
    assert "ACTIVITY (regime head)" in text, (
        "the scorecard does not report the activity head separately"
    )
    assert "vs majority class" in text, (
        "the scorecard does not report the direction head against its "
        "majority-class baseline"
    )
    assert "vs 50/50 baseline" in text, (
        "the scorecard does not report the activity head against its own "
        "baseline"
    )


# ---------------------------------------------------------------------------
# T12 -- the React app's default view and toggle
# ---------------------------------------------------------------------------

TOGGLE_TO_ADVANCED = "Advanced view"
TOGGLE_TO_SIMPLE = "Simple view"

# Panel titles carry `text-transform: uppercase` in the stylesheet, and
# WebDriver's `.text` returns what is *displayed* -- so every title match below
# is case-insensitive on purpose. The toggle's label is not transformed.
def _rendered_contains(rendered: str, needle: str) -> bool:
    return needle.lower() in rendered.lower()


# How long the app gets to fetch a payload and mount, per attempt.
BOOT_TIMEOUT_S = 45


def _dump_page(driver, paths, tag: str) -> Path | None:
    """
    Write what the browser was actually showing, for a failure to be read later.

    The browser cases are the only ones in the suite that depend on something
    outside the process (a real Chromium, a real socket). When one fails, the
    traceback says *that* an assertion failed but not *what the page looked
    like*, and that is the whole diagnosis. This is best-effort: nothing here
    may raise, because it runs while an exception is already propagating.
    """
    try:
        text = driver.find_element("css selector", "#root").text
    except Exception as exc:                                    # noqa: BLE001
        text = f"(could not read #root: {type(exc).__name__}: {exc})"
    try:
        source = driver.page_source
    except Exception as exc:                                    # noqa: BLE001
        source = f"(could not read page_source: {type(exc).__name__}: {exc})"

    body = (f"captured: {_support.now_iso()}\n"
            f"current url: {getattr(driver, 'current_url', '?')}\n"
            f"title: {getattr(driver, 'title', '?')}\n"
            + "=" * 78 + "\n#root text\n" + "=" * 78 + "\n" + text
            + "\n" + "=" * 78 + "\npage source\n" + "=" * 78 + "\n" + source)
    try:
        return _support.write_evidence(f"T12_failure_state_{tag}", body)
    except Exception:                                           # noqa: BLE001
        return None

# What the project's own SSR check asserts about the default view, and what
# each line means if it ever disappears.
DEFAULT_VIEW_CHECKS = {
    "opens on the simple view":
        "the app no longer opens on the simple view",
    "the advanced panels are not mounted":
        "the advanced panels are being mounted in the default view",
    "the toggle is offered":
        "no way to switch to the advanced view is being offered",
    "the price chart is present":
        "the default view no longer claims to draw the price chart",
    "the recorded history curve is claimed":
        "the default view no longer shows how earlier calls turned out",
    "the advanced-only layers are not claimed":
        "the default view is claiming the advanced view's chart layers",
}


def test_T12_the_app_opens_on_the_simple_view():
    """
    T12 -- run the project's own server-rendering check, ``web/check.mjs``,
    which renders the React components in Node and asserts the default view.

    Run as a subprocess rather than reimplemented: it is the project's own
    statement of what the default view is, and it is what has to keep passing.
    """
    if _support.node_exe() is None:
        pytest.skip("BLOCKED: node is not on PATH, so the React app cannot be "
                    "rendered at all")
    check = _support.WEB_DIR / "check.mjs"
    if not check.exists():
        pytest.skip(f"BLOCKED: {check} is missing")
    if not (_support.WEB_DIR / "node_modules").exists():
        pytest.skip(
            f"BLOCKED: {_support.WEB_DIR / 'node_modules'} is missing, so "
            "esbuild (which check.mjs bundles with) cannot be resolved. Run "
            "`npm install` in web/."
        )

    proc = _support.run_node(check, cwd=_support.WEB_DIR, timeout=300.0)
    output = (proc.stdout or "") + (proc.stderr or "")

    evidence = _support.write_evidence(
        "T12_react_default_view_check",
        f"$ node web/check.mjs   (cwd={_support.WEB_DIR})\n"
        f"exit code: {proc.returncode}\ncaptured: {_support.now_iso()}\n"
        + "=" * 78 + "\n" + output,
    )
    print(f"T12 evidence: {evidence}")

    assert proc.returncode == 0, (
        f"T12 FAILED: web/check.mjs exited {proc.returncode}.\n" + output[-4000:]
    )

    failed = [line for line in output.splitlines()
              if line.strip().startswith("FAIL")]
    assert not failed, (
        "T12 FAILED: web/check.mjs reported failing component(s):\n  "
        + "\n  ".join(failed)
    )

    missing = [f"{needle!r}: {why}"
               for needle, why in DEFAULT_VIEW_CHECKS.items()
               if needle not in output]
    assert not missing, (
        "T12 FAILED: the default-view block of web/check.mjs no longer asserts "
        "what T12 is about:\n  " + "\n  ".join(missing)
    )
    assert re.search(r"ok\s+opens on the simple view", output), (
        "the check lists 'opens on the simple view' but not as passing:\n"
        + output[-4000:]
    )
    assert "all components rendered" in output, (
        "web/check.mjs did not reach its success line; output was:\n"
        + output[-4000:]
    )


def test_T12_the_toggle_is_a_real_switch_and_remembers_the_choice():
    """
    T12 -- the toggle's mechanics, read out of ``web/src/App.jsx``: the
    ``localStorage`` key the choice is stored under, the guarded read that makes
    "no stored value" mean the simple view, and both directions of the switch.
    A structural check, so it runs without a browser.
    """
    app = (_support.WEB_DIR / "src" / "App.jsx").read_text(encoding="utf-8")

    assert "btc-dashboard-view" in app, (
        "the view choice is no longer persisted under a known localStorage key"
    )
    assert "useState(readStoredView)" in app, (
        "App no longer initialises the view from the stored value"
    )

    # The stored value must be read defensively: a corrupt entry, blocked site
    # data or the SSR check (which has no `window`) must all fall back to the
    # default rather than throwing on start-up.
    stored = re.search(r"function readStoredView\(\)\s*\{(.*?)\n\}", app,
                       re.DOTALL)
    assert stored, "readStoredView() is gone; the stored value is read inline"
    body = stored.group(1)
    assert "try" in body and "catch" in body, (
        "readStoredView() no longer guards its localStorage access; in a "
        "private window that accessor throws and the app would not start"
    )
    assert "'simple'" in body, (
        "readStoredView() no longer falls back to the simple view"
    )

    # Both directions of the toggle have to exist, or the switch is one-way.
    for label, why in ((TOGGLE_TO_ADVANCED, "reach the advanced view"),
                       (TOGGLE_TO_SIMPLE, "get back to the simple view")):
        assert label in app, f"the toggle offers no way to {why}"

    assert "SimpleView" in app and "advanced" in app, (
        "App.jsx no longer switches between a simple and an advanced view"
    )


def test_T12_the_built_bundle_is_served_and_the_toggle_switches_views(paths,
                                                                     captured_snapshot):
    """
    T12 -- the real thing: serve ``web/dist`` over loopback with the captured
    payload behind ``/api/snapshot`` and ``/api/stream``, open it in a headless
    browser, screenshot the default view, click the toggle, and screenshot the
    advanced view.

    Skipped, with the reason, when ``web/dist`` has not been built or no browser
    can be started -- never faked. If the page comes up but the toggle does not
    change the view, this fails rather than recording a pass.
    """
    if not paths.web_dist.exists():
        pytest.skip(
            f"BLOCKED: {paths.web_dist} does not exist, so there is no built "
            "app to open. Run `npm install && npm run build` in web/."
        )

    try:
        from test_api import _StubEngine        # the same engine stub T06 uses
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"BLOCKED: could not import the API engine stub: {exc}")

    from server import build_app

    engine = _StubEngine(captured_snapshot)
    app = build_app(engine, paths.web_dist)

    try:
        server_ctx = _support.serve_app(app)
        base = server_ctx.__enter__()
    except _support.ServerUnavailable as exc:
        pytest.skip(f"BLOCKED: no HTTP server available: {exc}")

    driver = None
    try:
        # -- the page must actually come up --------------------------------
        import httpx
        with httpx.Client(timeout=20.0) as client:
            index = client.get(f"{base}/")
            assert index.status_code == 200, (
                f"the built app returned {index.status_code} at / - the static "
                "mount is not serving index.html"
            )
            assert 'id="root"' in index.text, (
                "the served index.html has no React root element"
            )
            assert client.get(f"{base}/api/snapshot").status_code == 200

        # -- now drive it --------------------------------------------------
        try:
            driver = _support.open_browser()
        except _support.BrowserUnavailable as exc:
            pytest.skip(
                "PARTIAL: web/dist is served and answers 200, but no browser "
                f"could be started, so the default view could not be driven or "
                f"screenshotted: {exc}"
            )

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        def toggle_button(d):
            """The view toggle, or None while the app is still booting."""
            for b in d.find_elements(By.TAG_NAME, "button"):
                try:
                    if "view" in (b.text or "").lower():
                        return b
                except Exception:                               # noqa: BLE001
                    continue
            return None

        wait = WebDriverWait(driver, BOOT_TIMEOUT_S)

        def boot(attempt: str) -> None:
            """
            Load the app and wait for it to mount.

            The toggle only exists once a payload has arrived, so waiting for it
            is the real "the app is up" signal -- not just "the bundle loaded".

            Two waits, and both matter. The first is React having fetched and
            rendered a snapshot; the second is that render having produced a
            body of text, which is what the assertions below then measure.
            """
            driver.get(base)
            try:
                wait.until(lambda d: toggle_button(d) is not None)
                wait.until(lambda d: len(d.find_element(
                    By.CSS_SELECTOR, "#root").text) > 200)
            except Exception:                                   # noqa: BLE001
                _dump_page(driver, paths, attempt)
                raise

        try:
            boot("boot")
        except Exception as first:                              # noqa: BLE001
            # ONE reload, then no more. The app is deterministic given the
            # payload -- the same test mounts in a couple of seconds on its own
            # (the budget above is 45) -- so a first load that never mounts is a
            # browser/socket start-up race, not a property of the dashboard.
            # The retry is announced and its failure state is written to
            # evidence, so a reader can always see whether one happened; if the
            # second attempt fails too, this test fails.
            print(f"T12: first page load did not mount within "
                  f"{BOOT_TIMEOUT_S}s ({type(first).__name__}: "
                  f"{str(first).splitlines()[0][:120]}) - reloading once")
            boot("boot_retry")

        root_text = driver.find_element(By.CSS_SELECTOR, "#root").text
        label = toggle_button(driver).text.strip()
        assert TOGGLE_TO_ADVANCED in label, (
            f"the toggle reads {label!r} on start-up; it should offer the "
            f"advanced view ({TOGGLE_TO_ADVANCED!r}), i.e. the app opened on "
            "the simple one"
        )
        assert _rendered_contains(root_text,
                                  "Will the price move more than usual?"), (
            "the page opened with a payload but is not showing the simple "
            "view's headline answer"
        )
        assert not _rendered_contains(root_text, "held-out performance"), (
            "the advanced view's panels are mounted in the default view"
        )

        simple_png = _support.screenshot(
            driver, paths.screenshots / "T12_react_simple_view.png")
        (paths.screenshots / "T12_react_simple_view.txt").write_text(
            f"captured: {_support.now_iso()}\nurl: {base}\n"
            f"view: simple (default)\ntoggle label: {label!r}\n"
            f"rendered text: {len(root_text)} chars\n"
            + "=" * 78 + "\n" + root_text, encoding="utf-8")

        # -- click the toggle and confirm the view actually changed --------
        toggle_button(driver).click()

        def switched(d):
            btn = toggle_button(d)
            return btn is not None and TOGGLE_TO_SIMPLE in (btn.text or "")

        wait.until(switched)
        advanced_text = driver.find_element(By.CSS_SELECTOR, "#root").text
        new_label = toggle_button(driver).text.strip()

        advanced_png = _support.screenshot(
            driver, paths.screenshots / "T12_react_advanced_view.png")
        (paths.screenshots / "T12_react_advanced_view.txt").write_text(
            f"captured: {_support.now_iso()}\nurl: {base}\n"
            f"view: advanced (after clicking {label!r})\n"
            f"toggle label now: {new_label!r}\n"
            f"rendered text: {len(advanced_text)} chars\n"
            + "=" * 78 + "\n" + advanced_text, encoding="utf-8")

        assert advanced_text != root_text, (
            "clicking the toggle changed the button label but not the page"
        )
        assert len(advanced_text) > len(root_text), (
            f"the advanced view renders less text ({len(advanced_text)} chars) "
            f"than the simple one ({len(root_text)}); the panels do not appear "
            "to have mounted"
        )
        assert _rendered_contains(advanced_text, "held-out performance"), (
            "the advanced view is missing its held-out metrics panel"
        )
        assert _rendered_contains(advanced_text, "live direction scorecard"), (
            "the advanced view is missing its direction scorecard"
        )
        assert not _rendered_contains(advanced_text,
                                      "Will the price move more than usual?"), (
            "the simple view is still mounted underneath the advanced one"
        )

        for png in (simple_png, advanced_png):
            assert png.stat().st_size > 0, f"{png} is empty"
        print(f"T12 screenshots: {simple_png.name} "
              f"({simple_png.stat().st_size:,}B), {advanced_png.name} "
              f"({advanced_png.stat().st_size:,}B)")
    finally:
        if driver is not None:
            if sys.exc_info()[0] is not None:
                # Anything below the pre-flight failed while a browser was
                # open: keep what it was showing before it is torn down. This
                # runs while an exception is propagating, so _dump_page()
                # swallows its own errors.
                _dump_page(driver, paths, "failure_state")
            try:
                driver.quit()
            except Exception:                                   # noqa: BLE001
                pass
        server_ctx.__exit__(None, None, None)
