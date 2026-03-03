"""
tests/test_sanity.py
--------------------
Minimal sanity-check suite for the fairness_wrapper package.

Run with:
    pytest tests/test_sanity.py -v

Tests
-----
1. test_generate_shape          – DataFrame has 10 columns; every row sums to n.
2. test_ir_gr_in_unit_interval  – IR and GR are always in [0, 1].
3. test_eod_nan_policy          – Hand-crafted row: check each policy's output.
4. test_eod_range_when_defined  – Defined EOD values are in [−1, +1].
5. test_wrap_identity           – wrap() returns its input unchanged.
6. test_nan_rate_range          – nan_rate() is in [0, 1].
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from fairness_wrapper.generate import generate_cm_pairs
from fairness_wrapper.metrics import compute_eod, nan_rate
from fairness_wrapper.wrapper import wrap

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_N = 16  # fast enough for CI; ~1 820 rows


@pytest.fixture(scope="module")
def df_small() -> pd.DataFrame:
    """Generate a small dataset once per test session."""
    return generate_cm_pairs(n=SMALL_N)


@pytest.fixture(scope="module")
def df_with_eod(df_small: pd.DataFrame) -> pd.DataFrame:
    df = df_small.copy()
    df["eod"] = compute_eod(df, policy="nan")
    return df


# ---------------------------------------------------------------------------
# 1. Shape and row-sum
# ---------------------------------------------------------------------------

def test_generate_shape(df_small: pd.DataFrame):
    """DataFrame must have exactly 10 columns and every row must sum to SMALL_N."""
    assert df_small.shape[1] == 10, (
        f"Expected 10 columns (8 counts + ir + gr), got {df_small.shape[1]}."
    )

    cm_cols = ["i_tp", "i_fp", "i_tn", "i_fn", "j_tp", "j_fp", "j_tn", "j_fn"]
    row_sums = df_small[cm_cols].sum(axis=1)
    assert (row_sums == SMALL_N).all(), (
        f"Some rows do not sum to {SMALL_N}: {row_sums[row_sums != SMALL_N].head()}"
    )


# ---------------------------------------------------------------------------
# 2. IR and GR in [0, 1]
# ---------------------------------------------------------------------------

def test_ir_gr_in_unit_interval(df_small: pd.DataFrame):
    assert df_small["ir"].between(0.0, 1.0).all(), "IR out of [0, 1]."
    assert df_small["gr"].between(0.0, 1.0).all(), "GR out of [0, 1]."


# ---------------------------------------------------------------------------
# 3. NaN policy behaviour on a hand-crafted row
# ---------------------------------------------------------------------------

def _make_row(i_tp, i_fn, j_tp, j_fn) -> pd.DataFrame:
    """Return a single-row DataFrame with zero FP/TN to keep it minimal."""
    return pd.DataFrame(
        [
            {
                "i_tp": i_tp, "i_fp": 0, "i_tn": 0, "i_fn": i_fn,
                "j_tp": j_tp, "j_fp": 0, "j_tn": 0, "j_fn": j_fn,
            }
        ]
    )


class TestNanPolicy:
    """Tests for each NaN-handling policy."""

    def test_nan_policy_propagates_nan(self):
        """When i_tp + i_fn = 0, policy='nan' → EOD is NaN."""
        row = _make_row(i_tp=0, i_fn=0, j_tp=3, j_fn=1)
        eod = compute_eod(row, policy="nan")
        assert math.isnan(eod.iloc[0]), f"Expected NaN, got {eod.iloc[0]}"

    def test_zero_policy_fills_nan(self):
        """When i_tp + i_fn = 0, policy='zero' → EOD = TPR_j − 0 = TPR_j."""
        row = _make_row(i_tp=0, i_fn=0, j_tp=3, j_fn=1)
        eod = compute_eod(row, policy="zero")
        expected_tpr_j = 3 / (3 + 1)  # 0.75
        assert not math.isnan(eod.iloc[0]), "Expected a number, got NaN."
        assert abs(eod.iloc[0] - expected_tpr_j) < 1e-9, (
            f"Expected EOD={expected_tpr_j}, got {eod.iloc[0]}"
        )

    def test_smooth_alpha_no_nan(self):
        """smooth_alpha must never produce NaN, even when denominators are 0."""
        row = _make_row(i_tp=0, i_fn=0, j_tp=0, j_fn=0)
        eod = compute_eod(row, policy="smooth_alpha", alpha=1.0)
        assert not math.isnan(eod.iloc[0]), "smooth_alpha produced NaN."

    def test_smooth_alpha_value(self):
        """Verify the Laplace-smooth formula for a known case."""
        # i: TP=2, FN=2 → tpr_i = (2+1)/(4+2) = 3/6 = 0.5
        # j: TP=4, FN=0 → tpr_j = (4+1)/(4+2) = 5/6
        row = _make_row(i_tp=2, i_fn=2, j_tp=4, j_fn=0)
        eod = compute_eod(row, policy="smooth_alpha", alpha=1.0)
        expected = 5 / 6 - 3 / 6  # ≈ 0.3333
        assert abs(eod.iloc[0] - expected) < 1e-9, (
            f"Expected {expected:.6f}, got {eod.iloc[0]:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. EOD range when defined
# ---------------------------------------------------------------------------

def test_eod_range_when_defined(df_with_eod: pd.DataFrame):
    """All non-NaN EOD values must lie in [−1, +1]."""
    defined = df_with_eod["eod"].dropna()
    assert defined.between(-1.0 - 1e-9, 1.0 + 1e-9).all(), (
        f"EOD out of [-1,1]: min={defined.min():.4f}, max={defined.max():.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Wrapper identity
# ---------------------------------------------------------------------------

def test_wrap_identity():
    """wrap() must return its first argument unchanged (identity)."""
    cm_i = {"tp": 5, "fp": 2, "tn": 3, "fn": 1}
    cm_j = {"tp": 7, "fp": 1, "tn": 4, "fn": 2}
    for val in [0.42, -0.3, 0.0, float("nan")]:
        result = wrap(val, cm_i, cm_j, ir=0.5, gr=0.5, policy="nan")
        if math.isnan(val):
            assert math.isnan(result), f"Expected NaN, got {result}"
        else:
            assert result == val, f"Expected {val}, got {result}"


# ---------------------------------------------------------------------------
# 6. nan_rate range
# ---------------------------------------------------------------------------

def test_nan_rate_range():
    """nan_rate() must always return a value in [0, 1]."""
    for series in [
        pd.Series([1.0, 2.0, 3.0]),            # no NaNs  → 0.0
        pd.Series([np.nan, np.nan]),            # all NaNs → 1.0
        pd.Series([1.0, np.nan, 3.0]),          # mixed    → ~0.33
        pd.Series([], dtype=float),             # empty    → 0.0
    ]:
        rate = nan_rate(series)
        assert 0.0 <= rate <= 1.0, f"nan_rate={rate} out of [0,1] for {series.tolist()}"
