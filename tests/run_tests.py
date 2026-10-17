#!/usr/bin/env python
"""
Run the whole T01-T13 acceptance suite and write the run's artefacts.

    python tests/run_tests.py                 # everything
    python tests/run_tests.py -k T11          # extra args go straight to pytest
    python tests/run_tests.py --no-report     # no log/JSON (quick iteration)

What it writes, all inside ``tests/``:

    tests/logs/test_run.log    timestamped, per-test, per-case, then the full
                               console transcript of the run
    tests/logs/results.json    machine-readable: one verdict per case T01-T13,
                               every test's outcome, and an evidence index
    tests/screenshots/         the panel renders, page dumps and PNGs the
                               dashboard tests capture

The report files themselves are written by ``tests/conftest.py`` on every
pytest run -- including a bare ``pytest tests -v`` -- so this runner does not
duplicate that work. What it adds is the console transcript, the printed
T01-T13 table, and a check that the evidence was actually produced.

pytest runs in a child process, with this same interpreter, so the output
captured here is byte-for-byte what ``pytest tests -v`` prints and nothing in
this file can change how the tests themselves behave.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import _support  # noqa: E402


def _banner(command: str, extra: list[str]) -> str:
    return "\n".join([
        "=" * 78,
        "ACCEPTANCE TEST RUN - T01..T13",
        "=" * 78,
        f"started        : {_support.now_iso()}",
        f"command        : {command}",
        f"python         : {sys.version.split()[0]}  ({sys.executable})",
        f"node           : {_support.node_exe() or 'not found'}",
        f"project root   : {_support.PROJECT_ROOT}",
        f"pytest args    : {' '.join(extra) or '(none)'}",
        "=" * 78,
    ])


def run(extra: list[str] | None = None, *, verbose: bool = True,
        write_report: bool = True) -> tuple[int, str]:
    """
    Run pytest over ``tests/`` and return ``(exit_code, transcript)``.

    The child's stdout and stderr are echoed here as they arrive, and returned
    as one string so the caller (the notebook) can display it.
    """
    _support.ensure_dirs()

    argv = [sys.executable, "-m", "pytest", str(TESTS_DIR)]
    if verbose:
        argv.append("-v")
    argv += list(extra or ())

    env = dict(os.environ)
    # The evidence prints include the block glyphs the dashboard's sparkline
    # uses; a cp1252 console cannot encode them and would abort the run mid-way.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if not write_report:
        env["BTC_TESTS_NO_REPORT"] = "1"
    else:
        env.pop("BTC_TESTS_NO_REPORT", None)

    command = " ".join(argv)
    header = _banner(command, list(extra or ()))
    print(header, flush=True)

    chunks: list[str] = []
    proc = subprocess.Popen(
        argv,
        cwd=str(_support.PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            chunks.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        proc.wait()

    transcript = header + "\n" + "".join(chunks)
    return int(proc.returncode), transcript


def report(transcript: str, *, write_report: bool = True) -> dict | None:
    """
    Append the transcript to the run log and print the T01-T13 table.

    Returns the parsed results.json, or None when the report was suppressed.
    """
    if not write_report:
        print("report: suppressed (--no-report); nothing was written to "
              "tests/logs")
        return None

    # conftest.py wrote the structured log during the session; the raw console
    # transcript is appended after it so the file reads: structured first, then
    # the verbatim output that produced it.
    _support.append_run_log(
        "\n" + "=" * 78 + "\nCONSOLE TRANSCRIPT\n" + "=" * 78 + "\n"
        + transcript + "\n"
    )

    if not _support.RESULTS_JSON.exists():
        print(f"WARNING: {_support.RESULTS_JSON} was not written. Check that "
              "tests/conftest.py ran (it writes the report in "
              "pytest_sessionfinish).")
        return None

    results = _support.read_json(_support.RESULTS_JSON)
    table = _support.format_summary_table(results)
    print(table, flush=True)
    _support.append_run_log(
        f"\n{_support.now_iso()}  appended the console transcript to "
        f"{_support.RUN_LOG}\n")

    shots = results["evidence"]["screenshots"]
    pngs = [e for e in shots if e["path"].endswith(".png")]
    if not pngs:
        print("WARNING: no screenshots were produced. The dashboard tests "
              "(T11/T12) write them; check the skip reasons in results.json.")
    for entry in shots:
        print(f"  evidence  {entry['path']}  ({entry['bytes']:,}B)")
    print(f"\nlog      : {_support.RUN_LOG}")
    print(f"results  : {_support.RESULTS_JSON}")
    print(f"screenshots: {_support.SCREENSHOTS_DIR}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the T01-T13 acceptance suite and write its artefacts.")
    parser.add_argument("--no-report", action="store_true",
                        help="do not write tests/logs/test_run.log or "
                             "results.json")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="pass no -v to pytest")
    args, extra = parser.parse_known_args(argv)

    code, transcript = run(extra, verbose=not args.quiet,
                           write_report=not args.no_report)
    report(transcript, write_report=not args.no_report)

    print(f"\npytest exit code: {code}"
          + ("  (all tests passed)" if code == 0 else "  (see the failures "
             "above, and tests/logs/results.json)"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
