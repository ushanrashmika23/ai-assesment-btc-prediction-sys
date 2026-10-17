"""
Shared, TEST-ONLY helpers for the T01-T13 acceptance suite.

Nothing in this module is imported by production code under ``src/`` or
``web/``. Nothing here writes outside ``tests/`` -- the single exception is
the read-only HTTP stub in ``test_dashboard.py``, which binds an ephemeral
loopback port and touches no files at all.

External dependencies and how they are handled
----------------------------------------------
* **Node / npm** -- ``web/`` ships its own Node-based contract checks
  (``markers-check.mjs``, ``check.mjs``). They are the project's own tests of
  the JS layer and are invoked as subprocesses rather than reimplemented in
  Python, so a change to the JS contract fails *there* and is reported here.
* **A browser (Selenium + Edge)** -- needed only to rasterise the dashboard
  panels and the React app into screenshots. It is optional: when it is
  unavailable the tests still pass on their own assertions and the screenshot
  step records that it was skipped, with the reason. It is never faked.
* **Binance REST / TensorFlow** -- the live API integration test in
  ``test_api.py`` needs both. It is isolated in one test that skips with an
  explicit reason when either is unavailable.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Paths -- everything is derived from this file's location, so the suite runs
# from any working directory.
# ---------------------------------------------------------------------------

TESTS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = TESTS_DIR.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
WEB_DIR: Path = PROJECT_ROOT / "web"
NODE_DIR: Path = TESTS_DIR / "node"
DATA_FINAL: Path = PROJECT_ROOT / "data" / "final"
MODELS_DIR: Path = PROJECT_ROOT / "models"

SCREENSHOTS_DIR: Path = TESTS_DIR / "screenshots"
LOGS_DIR: Path = TESTS_DIR / "logs"

# The captured *real* payload the React project already uses as its own SSR
# fixture. Reusing it (rather than inventing one) means T06/T11/T12 exercise
# the shapes the live server actually emits.
CAPTURED_SNAPSHOT: Path = WEB_DIR / "_snapshot.json"
WEB_DIST: Path = WEB_DIR / "dist"

RESULTS_JSON: Path = LOGS_DIR / "results.json"
RUN_LOG: Path = LOGS_DIR / "test_run.log"


def ensure_dirs() -> None:
    """Create tests/screenshots and tests/logs. Safe to call repeatedly."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Local-clock ISO-8601 with an explicit UTC offset, second precision."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(fraction: float) -> str:
    return f"{100.0 * fraction:.1f}%"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def node_exe() -> str | None:
    """Path to node, or None when Node is not installed."""
    return shutil.which("node")


def run_node(
    script: str | Path,
    *,
    cwd: Path | None = None,
    timeout: float = 300.0,
    args: Iterable[str] = (),
) -> subprocess.CompletedProcess:
    """
    Run a Node script and capture its output.

    Raises RuntimeError (not AssertionError) when Node is missing, so the
    caller can decide between skipping and failing.
    """
    exe = node_exe()
    if exe is None:
        raise RuntimeError(
            "Node.js is not on PATH - the web/ contract checks cannot run. "
            "Install Node 18+ or skip the JS-backed tests."
        )
    script = Path(script)
    if not script.exists():
        raise RuntimeError(f"Node script not found: {script}")
    return subprocess.run(
        [exe, str(script), *args],
        cwd=str(cwd or script.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def run_python(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 900.0,
) -> subprocess.CompletedProcess:
    """Run a Python script with this interpreter, capturing both streams."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# A real HTTP server on an ephemeral loopback port
# ---------------------------------------------------------------------------

class ServerUnavailable(RuntimeError):
    """Raised when uvicorn is not installed, so no socket can be opened."""


@contextlib.contextmanager
def serve_app(app, *, host: str = "127.0.0.1", startup_timeout: float = 30.0):
    """
    Run an ASGI app on a free loopback port for the duration of the block.

    Yields ``http://127.0.0.1:<port>``.

    Why not ``fastapi.testclient.TestClient``? Because its transport buffers the
    whole response body before returning it (``portal.call(self.app, ...)``,
    then ``httpx.ByteStream(buffer.read())``), so it can never return from a
    route that streams forever. This project's ``/api/stream`` is exactly that,
    and what T08 has to verify is the bytes on the wire -- the ``event:`` and
    ``data:`` framing, the ``: keepalive`` heartbeat, the interval between
    frames. A real server is the only way to see them.

    Binds 127.0.0.1 only, on port 0, so it cannot collide with the dev server
    and is not reachable from off the machine. Touches no files.
    """
    try:
        import uvicorn
    except ImportError as exc:                                  # pragma: no cover
        raise ServerUnavailable(
            f"uvicorn is not installed ({exc}), so no HTTP server can be "
            "started. The tests that need one report as BLOCKED rather than "
            "passing on a substitute."
        ) from exc

    probe = socket.socket()
    try:
        probe.bind((host, 0))
        port = int(probe.getsockname()[1])
    finally:
        probe.close()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False, lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="test-uvicorn", daemon=True)

    started = time.monotonic()
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise ServerUnavailable(
                "the test HTTP server died during startup"
            )
        if time.monotonic() - started > startup_timeout:
            server.should_exit = True
            raise ServerUnavailable(
                f"the test HTTP server did not start within {startup_timeout}s"
            )
        time.sleep(0.02)

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20.0)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def write_evidence(name: str, text: str, *, directory: Path | None = None) -> Path:
    """
    Write a UTF-8 evidence artefact under tests/screenshots (default) or
    tests/logs, prefixed with a UTC timestamp so a re-run never hides an
    earlier one. Returns the path written.
    """
    target_dir = directory or SCREENSHOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{name}_{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_html_evidence(name: str, html: str, *, directory: Path | None = None) -> Path:
    target_dir = directory or SCREENSHOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{name}_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    return path


class BrowserUnavailable(RuntimeError):
    """Raised when a real browser cannot be started for screenshots."""


def open_browser(width: int = 1500, height: int = 1100):
    """
    Start a headless Edge browser. Selenium 4 fetches the matching driver
    itself (Selenium Manager), so no driver binary is committed to the repo.

    Raises BrowserUnavailable with the underlying reason rather than
    pretending a screenshot happened.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options as EdgeOptions
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BrowserUnavailable(f"selenium is not installed ({exc})") from exc

    options = EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--hide-scrollbars")
    options.add_argument(f"--window-size={width},{height}")
    # A deterministic device pixel ratio keeps screenshots comparable between
    # runs; the page layout does not depend on the host display.
    options.add_argument("--force-device-scale-factor=1")

    try:
        driver = webdriver.Edge(options=options)
    except Exception as exc:  # noqa: BLE001 - driver/network failures vary
        raise BrowserUnavailable(
            f"could not start Edge via Selenium: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        driver.set_window_size(width, height)
    except Exception:  # noqa: BLE001 - not fatal, the option already set it
        pass
    return driver


def screenshot(driver, path: Path) -> Path:
    """Save a PNG screenshot, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not driver.save_screenshot(str(path)):
        raise RuntimeError(f"driver refused to save a screenshot to {path}")
    return path


def stamp_png(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SCREENSHOTS_DIR / f"{name}_{stamp}.png"


# ---------------------------------------------------------------------------
# The T01-T13 case registry
# ---------------------------------------------------------------------------
# Titles are the acceptance criteria as stated in the project's test-case
# table. Every pytest node id in this suite carries its case id as the first
# token of the function name (``test_T07_...``), which is how results are
# grouped back into one verdict per case.

TEST_CASES: dict[str, str] = {
    "T01": "Leakage - corrupt every row after row 2,000, recompute features; "
           "earlier features unchanged (maximum difference = 0)",
    "T02": "Leakage - corrupt validation/test rows and refit the scaler; "
           "identical mean and scale",
    "T03": "Windows - every generated window checked for gap crossing "
           "(expected 0 crossings; total windows checked reported)",
    "T04": "Split - chronological ordering and purge boundaries "
           "(train < validation < test, no invalid overlap)",
    "T05": "Audit - all 24 existing data-quality/audit checks pass (24/24)",
    "T06": "API - snapshot, candles, prediction, track-record and metrics "
           "endpoints return HTTP 200 with valid response content",
    "T07": "API - candles?limit=5 returns HTTP 422 (minimum limit is 10)",
    "T08": "API - event stream opens and emits parseable snapshot/event frames",
    "T09": "Store - recording the exact same window twice creates no "
           "duplicate rows",
    "T10": "Store - opened read-only, a write is refused and the existing row "
           "count is unchanged",
    "T11": "Dashboard - every panel renders with real data and with empty "
           "data, without raising",
    "T12": "Dashboard - opens on the simple/default view and the toggle to "
           "the advanced view is available",
    "T13": "Markers - out-of-order calls are sorted, merged and never crash "
           "(10 calls -> 6 markers)",
}

CASE_ORDER: tuple[str, ...] = tuple(f"T{i:02d}" for i in range(1, 14))

# The case id is carried by the test function's name -- test_T07_limit_too_small
# -- so it is matched at the start of a name, not anywhere in the node id. A
# bare `\bT\d\d\b` would miss every id followed by an underscore (which is all
# of them) and could pick up an id out of a parametrisation instead.
_CASE_RE = re.compile(r"(?:^|[^A-Za-z0-9_])test_T(\d{2})(?![0-9])", re.IGNORECASE)
_CASE_ANY_RE = re.compile(r"(?:^|[^A-Za-z0-9_])T(\d{2})(?![0-9])", re.IGNORECASE)


def case_id_of(nodeid: str) -> str | None:
    """'tests/test_api.py::test_T07_limit_too_small' -> 'T07'."""
    match = _CASE_RE.search(nodeid) or _CASE_ANY_RE.search(nodeid)
    if not match:
        return None
    case = f"T{match.group(1)}"
    return case if case in TEST_CASES else None


def verdict_for(outcomes: list[str]) -> str:
    """
    One status per test case.

    PASS     every test for the case passed
    FAIL     at least one failed or errored
    BLOCKED  nothing ran - every test was skipped (reason is recorded)
    PARTIAL  some ran and passed, some were skipped (e.g. the optional
             screenshot half of a dashboard case)
    """
    if not outcomes:
        return "NOT IMPLEMENTED"
    if any(o in ("failed", "error") for o in outcomes):
        return "FAIL"
    n_skipped = sum(1 for o in outcomes if o == "skipped")
    if n_skipped == len(outcomes):
        return "BLOCKED"
    if n_skipped:
        return "PARTIAL"
    return "PASS"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def evidence_inventory() -> dict[str, list[dict]]:
    """Everything the run left in tests/screenshots and tests/logs."""
    out: dict[str, list[dict]] = {"screenshots": [], "logs": []}
    for key, directory in (("screenshots", SCREENSHOTS_DIR), ("logs", LOGS_DIR)):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            stat = path.stat()
            out[key].append({
                "path": _rel(path),
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            })
    return out


def build_results(
    *,
    started: str,
    finished: str,
    duration: float,
    records: list[dict],
    command: str,
    case_seconds: dict[str, float] | None = None,
    exit_code: int | None = None,
) -> dict:
    """
    Turn per-test records into results.json.

    ``case_seconds`` carries the wall-clock cost of each case over every phase.
    It is preferred over summing the records, which only hold the call phase and
    would under-report a case whose work happens in a module-scoped fixture.
    """
    by_case: dict[str, list[dict]] = {case: [] for case in CASE_ORDER}
    for rec in records:
        case = rec.get("case")
        if case in by_case:
            by_case[case].append(rec)

    cases = []
    for case in CASE_ORDER:
        recs = by_case[case]
        outcomes = [r["outcome"] for r in recs]
        measured = (case_seconds or {}).get(case)
        if measured is None:
            measured = sum(r.get("duration", 0.0) for r in recs)
        entry = {
            "id": case,
            "title": TEST_CASES[case],
            "status": verdict_for(outcomes),
            "n_tests": len(recs),
            "n_passed": sum(1 for o in outcomes if o == "passed"),
            "n_failed": sum(1 for o in outcomes if o in ("failed", "error")),
            "n_skipped": sum(1 for o in outcomes if o == "skipped"),
            "duration_seconds": round(measured, 3),
            "tests": [
                {
                    "nodeid": r["nodeid"],
                    "outcome": r["outcome"],
                    "duration_seconds": round(r.get("duration", 0.0), 3),
                    "skip_reason": r.get("skip_reason"),
                    "message": r.get("message"),
                }
                for r in recs
            ],
        }
        cases.append(entry)

    totals = {
        "n_tests": len(records),
        "passed": sum(1 for r in records if r["outcome"] == "passed"),
        "failed": sum(1 for r in records if r["outcome"] == "failed"),
        "errors": sum(1 for r in records if r["outcome"] == "error"),
        "skipped": sum(1 for r in records if r["outcome"] == "skipped"),
    }
    totals["cases_passed"] = sum(1 for c in cases if c["status"] == "PASS")
    totals["cases_failed"] = sum(1 for c in cases if c["status"] == "FAIL")
    totals["cases_blocked"] = sum(1 for c in cases if c["status"] == "BLOCKED")
    totals["cases_partial"] = sum(1 for c in cases if c["status"] == "PARTIAL")

    return {
        "suite": "BTCUSDT 15m trend prediction - acceptance suite (T01-T13)",
        "generated_utc": utc_iso(),
        "started": started,
        "finished": finished,
        "duration_seconds": round(duration, 3),
        "command": command,
        "exit_code": exit_code,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
            "node": node_exe() or None,
        },
        "project_root": _rel(PROJECT_ROOT),
        "totals": totals,
        "test_cases": cases,
        "tests": records,
        "evidence": evidence_inventory(),
    }


def format_summary_table(results: dict) -> str:
    """The T01-T13 table printed at the end of a run."""
    lines = [
        "",
        "=" * 78,
        "ACCEPTANCE SUITE - T01..T13",
        "=" * 78,
        f"{'ID':<5} {'STATUS':<10} {'#':>3} {'sec':>7}  TITLE",
        "-" * 78,
    ]
    for case in results["test_cases"]:
        lines.append(
            f"{case['id']:<5} {case['status']:<10} {case['n_tests']:>3} "
            f"{case['duration_seconds']:>7.2f}  {case['title'][:52]}"
        )
    lines.append("-" * 78)
    t = results["totals"]
    lines.append(
        f"tests: {t['n_tests']}  passed {t['passed']}  failed {t['failed']}  "
        f"errors {t['errors']}  skipped {t['skipped']}"
    )
    lines.append(
        f"cases: {t['cases_passed']} PASS  {t['cases_failed']} FAIL  "
        f"{t['cases_blocked']} BLOCKED  {t['cases_partial']} PARTIAL"
    )
    ev = results["evidence"]
    lines.append(
        f"evidence: {len(ev['screenshots'])} screenshot artefact(s) in "
        f"tests/screenshots, {len(ev['logs'])} in tests/logs"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def write_results(results: dict) -> Path:
    ensure_dirs()
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return RESULTS_JSON


def append_run_log(text: str) -> None:
    ensure_dirs()
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(text)


def reset_run_log(header: str) -> None:
    ensure_dirs()
    RUN_LOG.write_text(header, encoding="utf-8")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
