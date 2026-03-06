"""
tests/test_sanity.py
--------------------
Sanity-check suite for the experiments package.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from experiments.generate import generate_cm_pairs
from experiments.metrics  import compute_metric, compute_eod, nan_rate, perfect_fairness_rate
from experiments.wrapper  import apply_wrapper_df, wrap

SMALL_N = 16


@pytest.fixture(scope="module")
def df_small() -> pd.DataFrame:
    return generate_cm_pairs(n=SMALL_N)


def _row(**kw) -> pd.DataFrame:
    defaults = dict(i_tp=0, i_fp=0, i_tn=0, i_fn=0,
                    j_tp=0, j_fp=0, j_tn=0, j_fn=0)
    defaults.update(kw)
    return pd.DataFrame([defaults])


# ── generate ──────────────────────────────────────────────────────────────

def test_generate_shape(df_small):
    assert df_small.shape[1] == 10
    cm_cols = ["i_tp", "i_fp", "i_tn", "i_fn", "j_tp", "j_fp", "j_tn", "j_fn"]
    assert (df_small[cm_cols].sum(axis=1) == SMALL_N).all()


def test_ir_gr_in_unit_interval(df_small):
    assert df_small["ir"].between(0.0, 1.0).all()
    assert df_small["gr"].between(0.0, 1.0).all()


# ── classic metrics ────────────────────────────────────────────────────────

def test_eod_nan_policy():
    s = compute_eod(_row(i_tp=0, i_fn=0, j_tp=3, j_fn=1), policy="nan")
    assert math.isnan(s.iloc[0])


def test_eod_smooth_no_nan():
    s = compute_eod(_row(), policy="smooth_alpha", alpha=1.0)
    assert not math.isnan(s.iloc[0])


def test_eod_range(df_small):
    assert compute_eod(df_small, policy="nan").dropna().between(-1 - 1e-9, 1 + 1e-9).all()


def test_fprd_value():
    row = _row(i_tp=3, i_fp=1, i_tn=4, i_fn=2, j_tp=6, j_fp=2, j_tn=3, j_fn=1)
    assert abs(compute_metric(row, "fprd").iloc[0] - (2/5 - 1/5)) < 1e-9


def test_spd_value():
    row = _row(i_tp=3, i_fp=1, i_tn=4, i_fn=2, j_tp=6, j_fp=2, j_tn=3, j_fn=1)
    assert abs(compute_metric(row, "spd").iloc[0] - (8/12 - 4/10)) < 1e-9


def test_ppvd_zero_when_equal_ppv():
    row = _row(i_tp=3, i_fp=1, i_tn=4, i_fn=2, j_tp=6, j_fp=2, j_tn=3, j_fn=1)
    assert abs(compute_metric(row, "ppvd").iloc[0]) < 1e-9


def test_accd_value():
    row = _row(i_tp=3, i_fp=1, i_tn=4, i_fn=2, j_tp=6, j_fp=2, j_tn=3, j_fn=1)
    assert abs(compute_metric(row, "accd").iloc[0] - (9/12 - 7/10)) < 1e-9


def test_all_classic_no_nan_smooth(df_small):
    for m in ("eod", "fprd", "eqod", "spd", "ppvd", "npvd", "accd"):
        assert compute_metric(df_small, m, policy="smooth_alpha").isna().sum() == 0


def test_unknown_metric_raises(df_small):
    with pytest.raises(ValueError):
        compute_metric(df_small, "bogus_metric")


# ── eqod ──────────────────────────────────────────────────────────────────

def test_eqod_is_max_of_abs_eod_fprd():
    row = _row(i_tp=3, i_fp=1, i_tn=4, i_fn=2, j_tp=6, j_fp=2, j_tn=3, j_fn=1)
    eod  = abs(compute_metric(row, "eod").iloc[0])
    fprd = abs(compute_metric(row, "fprd").iloc[0])
    eqod = compute_metric(row, "eqod").iloc[0]
    assert abs(eqod - max(eod, fprd)) < 1e-9


def test_eqod_nonneg(df_small):
    assert (compute_metric(df_small, "eqod", policy="nan").dropna() >= -1e-9).all()


def test_eqod_leq_1(df_small):
    assert (compute_metric(df_small, "eqod", policy="nan").dropna() <= 1 + 1e-9).all()


def test_eqod_zero_when_both_zero():
    row = _row(i_tp=4, i_fp=2, i_tn=3, i_fn=1, j_tp=4, j_fp=2, j_tn=3, j_fn=1)
    assert abs(compute_metric(row, "eqod").iloc[0]) < 1e-9


def test_eqod_nan_propagates_correctly(df_small):
    eod_nan  = compute_metric(df_small, "eod",  policy="nan").isna().sum()
    fprd_nan = compute_metric(df_small, "fprd", policy="nan").isna().sum()
    eqod_nan = compute_metric(df_small, "eqod", policy="nan").isna().sum()
    assert eqod_nan >= max(eod_nan, fprd_nan)


# ── CAF metrics ────────────────────────────────────────────────────────────

def test_caf_no_nan(df_small):
    for m in ("caf_q_rms_ha", "caf_q_rms_ha_abs", "caf_eopp_q_ha"):
        assert compute_metric(df_small, m).isna().sum() == 0


def test_caf_finite_on_all_zeros():
    z = _row()
    for m in ("caf_q_rms_ha", "caf_q_rms_ha_abs", "caf_eopp_q_ha"):
        assert math.isfinite(compute_metric(z, m).iloc[0])


def test_caf_unsigned_nonneg(df_small):
    # caf_q_rms_ha is unsigned: ∈ [0, 1)
    s = compute_metric(df_small, "caf_q_rms_ha")
    assert (s >= -1e-9).all() and (s <= 1 + 1e-9).all()


def test_caf_unsigned_geq_abs_signed(df_small):
    # unsigned = sqrt(rms) of both Q values regardless of sign
    # signed = sign(Q0+Q1) * unsigned, so |signed| <= unsigned always
    # (when Q0+Q1=0, signed=0 but unsigned can be positive)
    unsigned = compute_metric(df_small, "caf_q_rms_ha").to_numpy()
    signed   = compute_metric(df_small, "caf_q_rms_ha_abs").abs().to_numpy()
    assert np.all(unsigned >= signed - 1e-9)


def test_caf_signed_range(df_small):
    s = compute_metric(df_small, "caf_q_rms_ha_abs")
    assert (s >= -1 - 1e-9).all() and (s <= 1 + 1e-9).all()


def test_caf_eopp_nonneg(df_small):
    s = compute_metric(df_small, "caf_eopp_q_ha")
    assert (s >= -1e-9).all() and (s <= 1 + 1e-9).all()


def test_caf_signed_antisymmetry():
    # swapping i and j should negate the signed CAF
    row_ij = _row(i_tp=4, i_fp=1, i_tn=9, i_fn=6, j_tp=8, j_fp=4, j_tn=16, j_fn=12)
    row_ji = _row(i_tp=8, i_fp=4, i_tn=16, i_fn=12, j_tp=4, j_fp=1, j_tn=9, j_fn=6)
    fwd = compute_metric(row_ij, "caf_q_rms_ha_abs").iloc[0]
    rev = compute_metric(row_ji, "caf_q_rms_ha_abs").iloc[0]
    assert abs(fwd + rev) < 1e-9


def test_caf_unsigned_symmetric():
    # unsigned should be equal regardless of which group is i vs j
    row_ij = _row(i_tp=4, i_fp=1, i_tn=9, i_fn=6, j_tp=8, j_fp=4, j_tn=16, j_fn=12)
    row_ji = _row(i_tp=8, i_fp=4, i_tn=16, i_fn=12, j_tp=4, j_fp=1, j_tn=9, j_fn=6)
    fwd = compute_metric(row_ij, "caf_q_rms_ha").iloc[0]
    rev = compute_metric(row_ji, "caf_q_rms_ha").iloc[0]
    assert abs(fwd - rev) < 1e-9


def test_caf_near_zero_when_proportional():
    prop = _row(i_tp=4, i_fp=2, i_tn=8, i_fn=6, j_tp=8, j_fp=4, j_tn=16, j_fn=12)
    assert compute_metric(prop, "caf_q_rms_ha").iloc[0] < 0.05
    assert compute_metric(prop, "caf_eopp_q_ha").iloc[0] < 0.05


def test_caf_kappa_changes_value():
    row = _row(i_tp=4, i_fp=1, i_tn=9, i_fn=6, j_tp=8, j_fp=4, j_tn=16, j_fn=12)
    v1 = compute_metric(row, "caf_q_rms_ha", kappa=0.5).iloc[0]
    v2 = compute_metric(row, "caf_q_rms_ha", kappa=1.0).iloc[0]
    assert v1 != v2


def test_caf_deprecated_alias(df_small):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = compute_metric(df_small, "caf")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    pd.testing.assert_series_equal(
        r.rename("caf_q_rms_ha"),
        compute_metric(df_small, "caf_q_rms_ha"),
    )


# ── Jeffreys-VST wrapper ───────────────────────────────────────────────────

CM_I = {"tp": 5, "fp": 2, "tn": 3, "fn": 1}
CM_J = {"tp": 7, "fp": 1, "tn": 4, "fn": 2}


def test_wrap_finite_degenerate():
    for deg in [{"tp":0,"fp":0,"tn":0,"fn":0}, {"tp":10,"fp":0,"tn":0,"fn":0}]:
        assert math.isfinite(wrap(deg, deg, metric="eod"))


def test_wrap_in_range():
    assert -1 - 1e-9 <= wrap(CM_I, CM_J, metric="eod") <= 1 + 1e-9


def test_wrap_antisymmetry():
    fwd = wrap(CM_I, CM_J, metric="eod")
    rev = wrap(CM_J, CM_I, metric="eod")
    assert abs(fwd + rev) < 1e-9


def test_wrap_zero_equal_groups():
    cm = {"tp":4,"fp":2,"tn":3,"fn":1}
    assert abs(wrap(cm, cm, metric="eod")) < 1e-9


def test_wrap_all_metrics():
    for m in ("eod", "fprd", "spd", "ppvd", "npvd", "accd"):
        assert math.isfinite(wrap(CM_I, CM_J, metric=m))


def test_apply_wrapper_df_shape(df_small):
    assert len(apply_wrapper_df(df_small, metric="eod")) == len(df_small)


def test_apply_wrapper_df_no_nans(df_small):
    assert apply_wrapper_df(df_small, metric="eod").isna().sum() == 0


def test_apply_wrapper_df_range(df_small):
    assert apply_wrapper_df(df_small, metric="eod").between(-1 - 1e-9, 1 + 1e-9).all()


# ── helpers ────────────────────────────────────────────────────────────────

def test_nan_rate():
    assert nan_rate(pd.Series([1.0, 2.0])) == 0.0
    assert nan_rate(pd.Series([np.nan, np.nan])) == 1.0
    assert nan_rate(pd.Series([], dtype=float)) == 0.0


def test_perfect_fairness_rate():
    s = pd.Series([-0.1, 0.0, 0.0, 0.2, np.nan])
    assert abs(perfect_fairness_rate(s, eps=0.0) - 2/4) < 1e-9
    assert abs(perfect_fairness_rate(s, eps=0.1) - 3/4) < 1e-9
