"""
Shared pytest configuration for the T01-T13 acceptance suite.

Three jobs:

1. **Make the project importable.** ``src/`` uses flat sibling imports
   (``from config import ...``), so ``src/`` must be on ``sys.path`` before any
   project module is imported. Done here once, for every entry point.

2. **Share fixtures.** The captured real payload, the evidence directory and
   the Node runner are session-scoped so a case can reuse another case's work
   without re-running it.

3. **Write the run report.** Any invocation of pytest over this directory --
   ``!pytest tests -v`` in the notebook, ``python tests/run_tests.py``, or a
   bare ``pytest tests`` from the terminal -- writes:

       tests/logs/test_run.log     structured, timestamped, per-test, per-case
       tests/logs/results.json     machine-readable results + evidence index

   Set ``BTC_TESTS_NO_REPORT=1`` to suppress both (useful while iterating).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest

# --- 1. imports ------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for _p in (str(SRC_DIR), str(PROJECT_ROOT), str(TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _support  # noqa: E402  (needs the sys.path edit above)


# --- 2. fixtures -----------------------------------------------------------

@pytest.fixture(scope="session")
def paths():
    """Namespace of the paths the suite reads and writes."""
    class Paths:
        tests = _support.TESTS_DIR
        root = _support.PROJECT_ROOT
        src = _support.SRC_DIR
        web = _support.WEB_DIR
        data_final = _support.DATA_FINAL
        models = _support.MODELS_DIR
        screenshots = _support.SCREENSHOTS_DIR
        logs = _support.LOGS_DIR
        captured_snapshot = _support.CAPTURED_SNAPSHOT
        web_dist = _support.WEB_DIST

    _support.ensure_dirs()
    return Paths


@pytest.fixture(scope="session")
def captured_snapshot() -> dict:
    """
    The real ``/api/snapshot`` payload captured by the project itself
    (``web/_snapshot.json``) and used by its own SSR checks.

    Reused here rather than invented, so the API and dashboard tests are
    asserted against the shape the live server actually emits. This is a
    *fixture file*, not a mock: no code path is replaced, only the input.
    """
    if not _support.CAPTURED_SNAPSHOT.exists():
        pytest.skip(
            f"captured payload missing: {_support.CAPTURED_SNAPSHOT} "
            "(re-capture with `python src/server.py --port 8099` then "
            "`curl -s http://127.0.0.1:8099/api/snapshot -o web/_snapshot.json`)"
        )
    return _support.read_json(_support.CAPTURED_SNAPSHOT)


@pytest.fixture(scope="session")
def final_dataset_available() -> bool:
    """True when ``python src/pipeline.py`` has been run at least once."""
    return (_support.DATA_FINAL / "dataset_metadata.json").exists()


# --- 3. reporting ----------------------------------------------------------

_SKIP_RE = re.compile(r"^Skipped: (.*)$", re.MULTILINE)

# How much of a failure to keep in results.json, and how much of it to keep from
# the END. pytest prints the source context first and the actual error last
# (the "E   ..." lines), so a plain head-truncation at a fixed budget drops
# exactly the lines a reader needs -- the assertion message. Keep both ends.
_LONGREPR_HEAD = 1200
_LONGREPR_TAIL = 2800


def _clip_longrepr(text) -> str | None:
    """A failure message for results.json, with the error lines preserved."""
    body = str(text or "")
    if not body:
        return None
    if len(body) <= _LONGREPR_HEAD + _LONGREPR_TAIL:
        return body
    dropped = len(body) - _LONGREPR_HEAD - _LONGREPR_TAIL
    return (body[:_LONGREPR_HEAD]
            + f"\n\n    ... {dropped:,} characters of traceback elided ...\n\n"
            + body[-_LONGREPR_TAIL:])


class _AcceptanceReporter:
    """
    Collects one record per test and writes results.json + test_run.log at the
    end of the session. Attached in ``pytest_configure``.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        # Wall-clock seconds per case over EVERY phase, not just the call
        # phase. A case whose work sits in a module-scoped fixture (T05 runs
        # the audit as a subprocess in one) does that work during setup, so
        # summing the recorded call-phase durations would report the audit as
        # taking 0.01s. Only the verdict comes from the call phase; the cost is
        # the whole setup+call+teardown.
        self.case_seconds: dict[str, float] = {}
        self._started_utc = _support.utc_iso()
        self._t0 = time.time()
        self._case_seen: set[str] = set()
        self._log: list[str] = []
        self._n_lines = 0

    # -- helpers ------------------------------------------------------------

    def _emit(self, line: str) -> None:
        """Timestamped line into the in-memory log, flushed at the end."""
        self._n_lines += 1
        self._log.append(f"{_support.utc_iso()}  {line}")

    @staticmethod
    def _skip_reason(report) -> str | None:
        text = getattr(report, "longrepr", None)
        if text is None:
            return None
        match = _SKIP_RE.search(str(text))
        return match.group(1).strip() if match else str(text).strip()[:400]

    # -- hooks --------------------------------------------------------------

    def pytest_runtest_logreport(self, report) -> None:
        case = _support.case_id_of(report.nodeid)

        if case:
            self.case_seconds[case] = self.case_seconds.get(case, 0.0) + float(
                getattr(report, "duration", 0.0) or 0.0
            )

        # The call phase carries the verdict; a non-passed setup (skip/error)
        # or teardown failure also has to be recorded, or a case could read as
        # PASS while its fixture never ran.
        if report.when == "call" or report.outcome != "passed":
            if report.passed:
                outcome = "passed"
            elif report.skipped:
                outcome = "skipped"
            elif report.failed and report.when == "setup":
                outcome = "error"
            else:
                outcome = "failed"

            message = None
            if outcome in ("failed", "error"):
                message = _clip_longrepr(getattr(report, "longrepr", ""))

            self.records.append({
                "nodeid": report.nodeid,
                "case": case,
                "outcome": outcome,
                "duration": float(getattr(report, "duration", 0.0) or 0.0),
                "when": report.when,
                "skip_reason": self._skip_reason(report) if outcome == "skipped" else None,
                "message": message,
                "logged_utc": _support.utc_iso(),
            })

            tag = f"[{case}]" if case else "[--]"
            if case and case not in self._case_seen:
                self._case_seen.add(case)
                self._emit(f"{tag} ===== case start =====")
            detail = ""
            if outcome == "skipped":
                detail = f"  reason: {self._skip_reason(report)}"
            self._emit(
                f"{tag} {outcome.upper():<7} "
                f"{report.duration * 1000:8.1f}ms  {report.nodeid}{detail}"
            )
            if message:
                for line in message.splitlines()[:40]:
                    self._emit(f"{tag}   | {line}")

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if _support.env_flag("BTC_TESTS_NO_REPORT"):
            return

        finished = _support.utc_iso()
        duration = time.time() - self._t0
        argv = " ".join(["pytest", *session.config.invocation_params.args])
        results = _support.build_results(
            started=self._started_utc,
            finished=finished,
            duration=duration,
            records=self.records,
            case_seconds=self.case_seconds,
            command=argv,
            exit_code=int(exitstatus),
        )

        header = "\n".join([
            "=" * 78,
            "ACCEPTANCE TEST RUN - T01..T13",
            "=" * 78,
            f"started        : {self._started_utc}",
            f"command        : {argv}",
            f"python         : {sys.version.split()[0]}  ({sys.executable})",
            f"project root   : {_support.PROJECT_ROOT}",
            "=" * 78,
            "",
        ])
        _support.reset_run_log(header)
        _support.append_run_log("\n".join(self._log) + "\n")
        _support.append_run_log(_support.format_summary_table(results) + "\n")

        _support.write_results(results)
        _support.append_run_log(
            f"\n{_support.utc_iso()}  wrote {_support.RESULTS_JSON}\n"
        )


def pytest_configure(config) -> None:
    """Register the case markers and attach the reporter."""
    for case, title in _support.TEST_CASES.items():
        config.addinivalue_line("markers", f"{case.lower()}: {title}")

    if not _support.env_flag("BTC_TESTS_NO_REPORT"):
        # One reporter per session; pytest_configure can fire more than once
        # when the suite is invoked programmatically.
        if not getattr(config, "_acceptance_reporter", None):
            config._acceptance_reporter = _AcceptanceReporter()
            config.pluginmanager.register(config._acceptance_reporter, "acceptance-report")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    """
    Fail loudly on a test whose name carries no T01-T13 case id: an
    unlabelled test would silently vanish from the summary.
    """
    unlabelled = [item.nodeid for item in items
                  if _support.case_id_of(item.nodeid) is None]
    if unlabelled:
        raise pytest.UsageError(
            "Every test in tests/ must carry its case id as the first token of "
            "the function name (test_T07_...). Unlabelled:\n  "
            + "\n  ".join(unlabelled)
        )


def pytest_report_header(config) -> str:
    node = _support.node_exe() or "not found"
    return (
        f"acceptance suite: T01-T13   project root: {_support.PROJECT_ROOT}\n"
        f"python: {sys.version.split()[0]}   node: {node}   "
        f"reports: {_support.RESULTS_JSON.name}, {_support.RUN_LOG.name}"
    )
