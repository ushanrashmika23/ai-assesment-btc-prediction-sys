"""
T05 -- run the project's own 24-check data-quality / leakage audit.

``src/audit_leakage.py`` is the project's adversarial self-examination: six
dynamic checks that re-execute the real feature, target, threshold and scaler
functions against a deliberately corrupted future, sixteen static checks over
the artefacts the pipeline actually wrote, and two regime-head checks. It is
run here **as a subprocess, exactly as the README documents it**, so what is
verified is the audit a reader would reproduce -- not a re-implementation of
it that could drift.

Why a subprocess rather than importing ``main()``:

* the audit is a script with module-level state (``RESULTS``) and its own exit
  code contract (0 iff every check passes) -- running it is the contract;
* its stdout is the evidence, and capturing it verbatim is the point.

Nothing is mocked. The audit reads ``data/`` read-only and writes nothing.
"""

from __future__ import annotations

import re

import pytest

import _support

# The audit prints one of these per check:
#   "  [PASS] 11a. Timestamp arrays are plain int64 (no pickled objects)"
_CHECK_RE = re.compile(r"^\s*\[(PASS|FAIL)\]\s+(\S+?)\s", re.MULTILINE)
_RESULT_RE = re.compile(r"AUDIT RESULT:\s*(\d+)/(\d+)\s+checks passed")
_FAILED_RE = re.compile(r"^\s*FAILED ->\s*(.+)$", re.MULTILINE)

# The 22 numbered checks plus the two lettered sub-checks the audit emits
# between them. Pinned so a check that silently stops running is caught by
# the *set*, not only by the count.
EXPECTED_LABELS = {f"{i}." for i in range(1, 23)} | {"11a.", "12a."}
EXPECTED_COUNT = 24


@pytest.fixture(scope="module")
def audit_run(final_dataset_available):
    """
    Run the audit once, module-scoped, and hand back its raw output.

    The audit re-reads a 60 MB feature matrix and the full split arrays, so
    running it per-test would triple the cost of the slowest case in the suite
    for no extra coverage.
    """
    if not final_dataset_available:
        pytest.skip(
            "data/final/dataset_metadata.json is missing, so the audit exits "
            "immediately with 'No final dataset found'. Run "
            "`python src/pipeline.py` first - this is a BLOCKED case, not a "
            "passing one."
        )

    proc = _support.run_python(
        [str(_support.SRC_DIR / "audit_leakage.py")],
        cwd=_support.PROJECT_ROOT,
        timeout=900.0,
    )
    output = (proc.stdout or "") + (proc.stderr or "")

    # Keep the transcript: it is the evidence for this case.
    evidence = _support.write_evidence(
        "T05_audit_transcript",
        f"$ python src/audit_leakage.py   (cwd={_support.PROJECT_ROOT})\n"
        f"exit code: {proc.returncode}\n"
        f"captured: {_support.now_iso()}\n"
        + "=" * 78 + "\n" + output,
    )
    print(f"T05 evidence: {evidence}")
    return proc, output


# ---------------------------------------------------------------------------
# T05
# ---------------------------------------------------------------------------

def test_T05_audit_reports_24_of_24_passed(audit_run):
    """
    T05 -- all 24 existing data-quality / audit checks pass.
    """
    proc, output = audit_run

    match = _RESULT_RE.search(output)
    assert match, (
        "the audit did not print an 'AUDIT RESULT: n/m checks passed' line at "
        "all, so it crashed before finishing.\n"
        f"exit code: {proc.returncode}\n"
        f"--- last 40 lines ---\n" + "\n".join(output.splitlines()[-40:])
    )

    n_pass, n_total = int(match.group(1)), int(match.group(2))

    assert n_total == EXPECTED_COUNT, (
        f"the audit ran {n_total} checks, but this case is defined as the "
        f"project's {EXPECTED_COUNT} checks - either a check was added "
        "(update EXPECTED_COUNT and the case table) or one silently stopped "
        "running"
    )
    assert n_pass == n_total, (
        f"T05 FAILED: {n_pass}/{n_total} audit checks passed.\n"
        "Failed checks:\n  "
        + "\n  ".join(_FAILED_RE.findall(output) or ["(unparsed - see the "
                                                    "transcript in tests/screenshots/)"])
    )
    assert proc.returncode == 0, (
        f"the audit reported {n_pass}/{n_total} but exited {proc.returncode}; "
        "it is documented to exit non-zero if any check fails"
    )


def test_T05_all_24_checks_are_present_and_named(audit_run):
    """
    T05 -- the checks that ran are the checks that are supposed to run.

    A count alone would not catch a refactor that replaced one check with
    another while keeping the total at 24.
    """
    _proc, output = audit_run

    seen = _CHECK_RE.findall(output)
    labels = [label for _flag, label in seen]
    failures = [label for flag, label in seen if flag == "FAIL"]

    assert len(labels) == EXPECTED_COUNT, (
        f"expected {EXPECTED_COUNT} [PASS]/[FAIL] lines, found {len(labels)}: "
        f"{labels}"
    )
    assert not failures, f"audit check(s) FAILED: {failures}"

    missing = EXPECTED_LABELS - set(labels)
    unexpected = set(labels) - EXPECTED_LABELS
    assert not missing, f"audit check(s) did not run: {sorted(missing)}"
    assert not unexpected, f"unexpected audit check label(s): {sorted(unexpected)}"
    assert len(set(labels)) == len(labels), (
        "the audit printed a duplicate check label, so "
        f"{len(labels) - len(set(labels))} check(s) ran twice: "
        f"{sorted({x for x in labels if labels.count(x) > 1})}"
    )


def test_T05_the_audit_actually_examines_the_leakage_questions(audit_run):
    """
    T05 -- the audit is not passing vacuously.

    Each of these four checks is the audit's own answer to one of the questions
    T01-T04 ask. Their presence and their *detail* lines are asserted, because
    a check whose body was emptied would still print ``[PASS]``.
    """
    _proc, output = audit_run

    expectations = {
        "1.":  ["max |feature diff| on rows < 2000 = 0.000e+00"],
        "4.":  ["identical mean_ and scale_"],
        "11.": ["required > 69 rows", "lookback 64", "purge=70"],
        "20.": ["0 crossing", "windows checked"],
        "22.": ["deviation from 50/50"],
    }
    for label, needles in expectations.items():
        check_re = re.compile(
            rf"\[PASS\]\s+{re.escape(label)}\s+(.*?)(?=\n\s*\n|\n\s*\[PASS\]|\n\s*\[FAIL\]|\n=)",
            re.DOTALL,
        )
        block = check_re.search(output)
        assert block, f"audit check {label} did not run or did not pass"
        body = block.group(1)
        for needle in needles:
            assert needle.lower() in body.lower(), (
                f"audit check {label} passed but its detail no longer "
                f"mentions {needle!r}, so it may no longer be checking what "
                f"it claims.\n  detail: {body.strip()[:400]}"
            )


def test_T05_the_shipped_dataset_the_audit_grades_is_the_one_present(paths):
    """
    T05 -- tie the audit to the artefacts: the metadata the audit reads must
    describe the array shapes actually on disk. An audit passing against a
    stale metadata file would be an audit of a different dataset.
    """
    import json

    import numpy as np

    meta_path = paths.data_final / "dataset_metadata.json"
    if not meta_path.exists():
        pytest.skip("data/final/dataset_metadata.json not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shapes = meta["array_shapes"]

    for key in ("X_train", "X_val", "X_test"):
        path = paths.data_final / f"{key}.npy"
        assert path.exists(), f"{path.name} is missing but metadata names it"
        on_disk = list(np.load(path, mmap_mode="r").shape)
        assert on_disk == list(shapes[key]), (
            f"{key}.npy on disk is {on_disk} but dataset_metadata.json "
            f"declares {shapes[key]}"
        )

    assert meta["dataset"]["n_features"] == shapes["X_train"][1], (
        "the declared feature count does not match the training block's width"
    )
    assert meta["splits"]["purge_samples"] == 70, (
        "PURGE_SAMPLES recorded in the metadata is no longer 70; the audit's "
        "check 11 arithmetic depends on it"
    )
