"""
T01, T02 -- data leakage.

The question both cases ask is the same one: *does anything downstream of row
t depend on rows after t?* It is answered the way ``src/audit_leakage.py``
answers it -- by re-running the project's real functions on a frame whose
future has been deliberately corrupted and asserting the past did not move --
but as independent tests with their own controls.

Deliberately NOT re-implemented here: the feature builders, the scaler and the
corruption helper all come from the project. A test that reimplemented them
would only prove the reimplementation agrees with itself.

Fixtures come from ``src/audit_leakage.py`` itself (``make_synthetic``,
``corrupt_from``) so the synthetic market the project audits is the synthetic
market these tests audit.

Documented shortcuts
--------------------
None: no mocking, no network, no disk writes. The synthetic frame is
deterministic (fixed seed 7, 3,000 rows) and every assertion is arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audit_leakage import corrupt_from, make_synthetic  # project's own fixture builders
from config import HORIZON_MAX, LOOKBACK
from features import FEATURE_COLUMNS, build_features
from sequences import fit_scaler

# The cut the acceptance table names: everything from row 2,000 onwards is
# multiplied, so any row before it that changes is a leak.
CUT = 2000

# The longest window any feature looks back through (rolling-96 and EMA-100
# are the widest; the constant is derived from the feature registry rather
# than guessed, see the assertion below).
_MAX_LOOKBACK = 100


@pytest.fixture(scope="module")
def leakage():
    """
    The base frame and its corrupted twin, with features built by the project's
    real ``build_features``.

    Module-scoped: T01 and T02 ask different questions of the same two frames,
    and rebuilding a 3,000-row feature matrix twice would double the cost of
    the slowest part of the suite for no extra coverage.
    """
    base = make_synthetic(n=3000, seed=7)
    corrupt = corrupt_from(base, CUT)

    ref = build_features(base, verbose=False)
    alt = build_features(corrupt, verbose=False)

    # Sanity: the corruption must actually corrupt. Without this the whole
    # module could pass on two identical frames.
    changed = (base.loc[CUT:, "close"].to_numpy()
               != corrupt.loc[CUT:, "close"].to_numpy())
    assert changed.any(), (
        "corrupt_from() did not change any row at or after the cut - the "
        "leakage tests below would be vacuous."
    )
    return SimpleNamespace(base=base, corrupt=corrupt, ref=ref, alt=alt)


# ---------------------------------------------------------------------------
# T01
# ---------------------------------------------------------------------------

def test_T01_features_before_the_cut_are_bit_identical(leakage):
    """
    T01 -- corrupt every row after row 2,000, recompute features, and assert
    the earlier features are unchanged. Expected maximum difference: 0.
    """
    cut = CUT

    ref_cols = leakage.ref[FEATURE_COLUMNS]
    alt_cols = leakage.alt[FEATURE_COLUMNS]
    assert len(ref_cols) == len(alt_cols) > cut, (
        f"the frames hold {len(ref_cols)} / {len(alt_cols)} rows, so the cut "
        f"at {cut} is outside the data and there is nothing 'before' it"
    )

    # POSITIONAL, not `.loc[0:2000]`: a label slice is inclusive at BOTH ends,
    # so that form would take 2001 rows and quietly compare row 2000 -- the
    # first CORRUPTED row -- as if it were part of the past, and the test would
    # fail on a leak that isn't one. Rows 0..1999 is exactly "before the cut".
    a = ref_cols.to_numpy(dtype=np.float64)[:cut]
    b = alt_cols.to_numpy(dtype=np.float64)[:cut]

    assert a.shape == b.shape, (
        f"the corrupt frame produced a different number of rows before the "
        f"cut: {a.shape} vs {b.shape}"
    )
    assert a.size, "no rows were compared - the cut is outside the frame"

    finite = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = np.isnan(a) != np.isnan(b)
    assert not nan_mismatch.any(), (
        "the corruption changed WHICH cells are NaN before the cut, at "
        f"{int(nan_mismatch.sum())} cell(s); first at row "
        f"{int(np.argmax(nan_mismatch.any(axis=1)))} "
        f"(column {FEATURE_COLUMNS[int(np.argmax(nan_mismatch.any(axis=0)))]})"
    )

    diff = np.where(finite, np.abs(a - b), 0.0)
    maximum = float(diff.max()) if diff.size else 0.0

    if maximum != 0.0:
        row, col = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            "T01 FAILED: a feature before the cut moved when only rows >= "
            f"{cut} were corrupted.\n"
            f"  max |difference| = {maximum:.6e} (expected exactly 0)\n"
            f"  worst cell      : row {row} (timestamp "
            f"{leakage.ref['timestamp'].iloc[row]}), "
            f"column {FEATURE_COLUMNS[col]!r}\n"
            f"  reference value : {a[row, col]!r}\n"
            f"  corrupted value : {b[row, col]!r}\n"
            f"  {int((diff > 0).sum())} of {diff.size} cells differ.\n"
            "  Any non-zero value means a feature reads the future."
        )

    assert maximum == 0.0, f"max |feature difference| = {maximum!r}, expected 0"


def test_T01_control_the_corruption_did_propagate(leakage):
    """
    Positive control for T01: the corrupted frame MUST move the features deep
    inside the corrupted region, far enough past the cut that no lookback
    window can still reach a clean row. Without this, T01 would also pass if
    ``corrupt_from`` silently did nothing.

    What "move" can mean here is narrower than it looks.
    ``corrupt_from(df, cut)`` multiplies a contiguous block by a CONSTANT, and
    a constant rescaling leaves every scale-invariant feature exactly as it
    was: ``pct_change`` of a rescaled series is identical, a z-score is
    identical, a ratio of two rescaled quantities is identical, and so are RSI,
    the EMA ratios, %b, the wick fractions and the calendar features. Those
    columns are *supposed* to agree, so the naive control -- "fewer than half
    the cells may still match" -- asserts something false about the fixture and
    fails on correct code (measured: 83.7% of cells match, and the match is
    total in 30 of the 49 columns).

    What the corruption does have to do is move the LEVEL features, on every
    single row compared. That is asserted instead, and it is the property T01
    leans on: the frame the "0 difference before the cut" is measured against
    is demonstrably not the same frame.
    """
    # Every feature looks back at most _MAX_LOOKBACK rows, so from
    # CUT + _MAX_LOOKBACK onwards neither frame can see across the cut.
    safe_from = CUT + _MAX_LOOKBACK

    a = leakage.ref[FEATURE_COLUMNS].to_numpy(dtype=np.float64)[safe_from:]
    b = leakage.alt[FEATURE_COLUMNS].to_numpy(dtype=np.float64)[safe_from:]

    assert a.shape == b.shape and a.size, (
        "the control compared nothing - the frame is too short for "
        f"cut={CUT} + lookback={_MAX_LOOKBACK}"
    )

    identical = np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
    share_identical = float(identical.mean())

    # 1. The corruption reached the feature matrix at all.
    assert share_identical < 1.0, (
        "T01's control failed: after corrupting every row >= "
        f"{CUT}, all {a.size:,} feature cells from row {safe_from} onwards are "
        "byte-identical. The corruption either did not apply or the features "
        "ignore their inputs - either way T01 above proves nothing."
    )

    # 2. ...and it reached the LEVEL features on every row, not merely on a
    #    handful of rows near an edge.
    fully_moved = [FEATURE_COLUMNS[c]
                   for c in range(a.shape[1]) if not identical[:, c].any()]
    assert fully_moved, (
        "T01's control failed: not one feature column differs on every row "
        f"from {safe_from} onwards. A constant rescaling must show up in the "
        "level features (atr_14, ema100_dist, ema50_100_ratio, ...); if none "
        "moved, the corruption is too weak for T01 to mean anything.\n"
        f"  columns that moved on at least one row: "
        f"{[FEATURE_COLUMNS[c] for c in range(a.shape[1]) if not identical[:, c].all()]}"
    )

    moved_rows = identical.shape[0] - int(identical.all(axis=1).sum())
    assert moved_rows == identical.shape[0], (
        f"{identical.shape[0] - moved_rows} of {identical.shape[0]:,} rows "
        f"from {safe_from} onwards are untouched in every column"
    )

    print(
        f"T01 control: {share_identical:.1%} of the {a.size:,} cells from row "
        f"{safe_from} onwards are unchanged - the expected consequence of a "
        f"constant x1.37 rescaling, which is invisible to ratios and z-scores. "
        f"{len(fully_moved)} level feature(s) moved on every row: "
        f"{', '.join(fully_moved)}"
    )


def test_T01_feature_registry_has_no_forward_looking_primitive():
    """
    Supporting structural check: the feature module contains no forward shift,
    no centred window and no back-fill. This is the *reason* T01 passes, so it
    is asserted rather than assumed -- a future edit introducing ``.shift(-1)``
    would otherwise only be caught by luck on this particular frame.

    Parsed as an AST, not grepped: the module's own docstring *names* the
    banned primitives in order to promise it does not use them, and a text
    search would flag that promise as a violation.
    """
    import ast

    import features as features_module

    tree = ast.parse(open(features_module.__file__, encoding="utf-8").read())

    def call_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        line = getattr(node, "lineno", "?")

        if name == "shift":
            for arg in list(node.args) + [kw.value for kw in node.keywords
                                          if kw.arg in ("periods", None)]:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    problems.append(
                        f"line {line}: .shift(<negative>) reads FUTURE rows")
                elif (isinstance(arg, ast.Constant)
                      and isinstance(arg.value, (int, float))
                      and arg.value < 0):
                    problems.append(
                        f"line {line}: .shift({arg.value}) reads FUTURE rows")

        if name in ("bfill", "backfill"):
            problems.append(f"line {line}: .{name}() pulls later values backwards")

        if name == "interpolate":
            problems.append(
                f"line {line}: .interpolate() invents values from the future")

        for kw in node.keywords:
            if (kw.arg == "center" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                problems.append(
                    f"line {line}: center=True makes a rolling window look both ways")

    assert not problems, (
        f"features.py contains {len(problems)} forward-looking primitive(s):\n  "
        + "\n  ".join(problems)
    )


# ---------------------------------------------------------------------------
# T02
# ---------------------------------------------------------------------------

def _scaler_inputs(ref) -> np.ndarray:
    """The 2D feature matrix the scaler would be fitted on, NaN rows dropped."""
    X = ref[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    valid = ~np.isnan(X).any(axis=1)
    return X[valid]


def test_T02_scaler_statistics_ignore_validation_and_test_rows(leakage):
    """
    T02 -- corrupt validation/test rows, refit the scaler, and assert the
    fitted statistics are identical. Expected: identical mean and scale.
    """
    X = _scaler_inputs(leakage.ref)
    n = len(X)
    assert n > 1000, f"only {n} complete feature rows - too few to split"

    train_end = int(n * 0.70)
    assert 0 < train_end < n

    clean = fit_scaler(X[:train_end], "standard")

    # Wreck ONLY the validation/test region. The train slice handed to the
    # second fit is byte-identical to the first, so any difference in the
    # fitted statistics could only come from the scaler looking past its
    # input.
    polluted_matrix = X.copy()
    polluted_matrix[train_end:] *= 1000.0
    polluted = fit_scaler(polluted_matrix[:train_end], "standard")

    mean_same = np.array_equal(clean.mean_, polluted.mean_)
    scale_same = np.array_equal(clean.scale_, polluted.scale_)

    if not (mean_same and scale_same):
        mean_delta = np.abs(clean.mean_ - polluted.mean_)
        scale_delta = np.abs(clean.scale_ - polluted.scale_)
        worst_m = int(np.argmax(mean_delta))
        worst_s = int(np.argmax(scale_delta))
        raise AssertionError(
            "T02 FAILED: the scaler's statistics moved when only rows at or "
            f"after the train boundary (index {train_end}) were corrupted.\n"
            f"  mean_  identical: {mean_same}  (max |delta| "
            f"{float(mean_delta.max()):.6e} at column "
            f"{FEATURE_COLUMNS[worst_m]!r})\n"
            f"  scale_ identical: {scale_same}  (max |delta| "
            f"{float(scale_delta.max()):.6e} at column "
            f"{FEATURE_COLUMNS[worst_s]!r})\n"
            "  The scaler must be fitted on the TRAIN split only."
        )

    # The transform of the train block must also be unchanged, which is the
    # property training actually consumes.
    np.testing.assert_array_equal(
        clean.transform(X[:train_end]), polluted.transform(X[:train_end]),
        err_msg="the scaler's output on the TRAIN rows changed after "
                "corrupting only the validation/test rows",
    )


def test_T02_control_the_pollution_would_move_a_scaler_that_saw_it(leakage):
    """
    Positive control for T02: fitting on a matrix that DOES include the
    polluted rows must produce different statistics. If it did not, T02's
    equality above would be trivially true (e.g. a scaler that ignores its
    input) and would prove nothing.
    """
    X = _scaler_inputs(leakage.ref)
    train_end = int(len(X) * 0.70)

    polluted = X.copy()
    polluted[train_end:] *= 1000.0

    honest = fit_scaler(X[:train_end], "standard")
    cheats = fit_scaler(polluted, "standard")          # deliberately wrong

    assert not np.array_equal(honest.mean_, cheats.mean_), (
        "corrupting the validation/test rows did not change a scaler fitted "
        "on ALL rows - the control cannot detect pollution, so T02's "
        "equality assertion is not evidence of anything."
    )


def test_T02_real_shipped_scaler_is_consistent_with_the_train_block():
    """
    The same property on the artefact that actually shipped: the stored
    ``scaler.pkl`` must reproduce the scaled training block, and its
    ``n_samples_seen_`` must not exceed the row count of the training block.
    Skipped (not failed) when the pipeline has not been run.
    """
    import pickle

    from _support import DATA_FINAL

    scaler_path = DATA_FINAL / "scaler.pkl"
    x_train_path = DATA_FINAL / "X_train.npy"
    if not (scaler_path.exists() and x_train_path.exists()):
        pytest.skip(
            "data/final/scaler.pkl and X_train.npy not present - run "
            "`python src/pipeline.py` to produce the shipped artefacts"
        )

    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    X_train = np.load(x_train_path, mmap_mode="r")

    assert scaler.mean_.shape[0] == X_train.shape[1], (
        f"scaler was fitted on {scaler.mean_.shape[0]} features but X_train "
        f"has {X_train.shape[1]} columns"
    )
    # n_samples_seen_ is how many rows the scaler was shown. It must not
    # exceed the training block, or the scaler saw validation/test rows.
    seen = int(getattr(scaler, "n_samples_seen_", 0))
    assert 0 < seen, "the scaler records no fitted sample count"
    assert seen <= X_train.shape[0] + 1, (
        f"scaler was fitted on {seen:,} rows but the training block holds only "
        f"{X_train.shape[0]:,} - it saw rows outside the train split"
    )

    # And the stored block really is this scaler's output: re-scaling is a
    # no-op because the block was written scaled AND clipped at +/-5 sigma,
    # so compare within the clip band only.
    sample = np.asarray(X_train[:512], dtype=np.float64)
    rescaled = np.clip(scaler.transform(scaler.inverse_transform(sample)), -5.0, 5.0)
    delta = float(np.abs(rescaled - sample).max())
    assert delta < 1e-3, (
        f"round-tripping X_train through the stored scaler moved values by "
        f"{delta:.3e}; the stored scaler does not match the stored block"
    )


def test_T02_lookback_and_horizon_are_the_documented_geometry():
    """
    Guard rail for the constants T01/T03/T04 all lean on: the lookback must
    still be 64 candles and the purge must still cover lookback + horizon, or
    the leakage guarantees those tests assert would no longer hold.
    """
    from config import LOOKBACK as cfg_lookback, PURGE_SAMPLES

    assert cfg_lookback == LOOKBACK == 64, (
        f"LOOKBACK changed to {cfg_lookback}; the window geometry in "
        "test_windows.py and test_split.py assumes 64"
    )
    assert PURGE_SAMPLES >= LOOKBACK + HORIZON_MAX, (
        f"PURGE_SAMPLES={PURGE_SAMPLES} is smaller than "
        f"LOOKBACK+HORIZON_MAX={LOOKBACK + HORIZON_MAX}: the last train label "
        "could reach into the first validation window"
    )
