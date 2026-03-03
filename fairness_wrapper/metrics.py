"""
metrics.py
----------
Computation of fairness metrics with explicit NaN-handling policies.

Currently implements:
  * Equal Opportunity Difference (EOD) = TPR_j − TPR_i

NaN policies
------------
"nan"
    Standard floating-point division.  When a denominator is zero the
    result is NaN (via pandas' default behaviour).  This is the most
    mathematically honest option.

"zero"
    If a group's true-positive rate is undefined (TP+FN == 0), treat that
    rate as 0.  This is a modelling convention: a group with no actual
    positives cannot be evaluated for recall; we impute recall = 0.
    **Note:** this biases EOD towards −1 when group *i* has no positives
    but group *j* does, or towards +1 in the reverse case.

"smooth_alpha"
    Laplace / add-alpha smoothing applied *before* division:
        TPR_g = (TP_g + α) / (TP_g + FN_g + 2α)
    This keeps every rate strictly between 0 and 1 and is the standard
    Bayesian-smoothing approach used in NLP.  α=1 is the Laplace prior.
    The denominator can never be 0 (α > 0), so no NaNs occur.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

NanPolicy = Literal["nan", "zero", "smooth_alpha"]

_VALID_POLICIES: tuple[str, ...] = ("nan", "zero", "smooth_alpha")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tpr(tp: pd.Series, fn: pd.Series,
         policy: NanPolicy, alpha: float) -> pd.Series:
    """Compute TPR for one group under the chosen NaN policy."""
    denom = tp + fn
    if policy == "nan":
        # pandas issues 0/0 = NaN naturally with float division
        return tp / denom.where(denom != 0, other=np.nan)
    elif policy == "zero":
        return (tp / denom.where(denom != 0, other=np.nan)).fillna(0.0)
    elif policy == "smooth_alpha":
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0 for smooth_alpha policy; got {alpha}")
        return (tp + alpha) / (denom + 2 * alpha)
    else:
        raise ValueError(
            f"Unknown NaN policy '{policy}'. Choose from {_VALID_POLICIES}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_eod(
    df: pd.DataFrame,
    policy: NanPolicy = "nan",
    alpha: float = 1.0,
) -> pd.Series:
    """Compute Equal Opportunity Difference for every row in *df*.

    EOD = TPR_j − TPR_i
    where TPR_g = TP_g / (TP_g + FN_g).

    A positive value means the *unprotected* group (j) has higher recall.
    A value of 0 means equal opportunity (perfect fairness for this metric).
    The theoretical range when defined is [−1, +1].

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns i_tp, i_fn, j_tp, j_fn (at minimum).
    policy : {"nan", "zero", "smooth_alpha"}
        How to handle undefined TPR (denominator = 0).  See module docstring.
    alpha : float
        Smoothing strength for the "smooth_alpha" policy (default 1.0).

    Returns
    -------
    pd.Series (float64)
        EOD values aligned to *df*'s index.  May contain NaN if
        policy == "nan" and any denominator is zero.
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"Unknown NaN policy '{policy}'. Choose from {_VALID_POLICIES}."
        )

    tpr_i = _tpr(df["i_tp"], df["i_fn"], policy=policy, alpha=alpha)
    tpr_j = _tpr(df["j_tp"], df["j_fn"], policy=policy, alpha=alpha)

    eod = tpr_j - tpr_i
    eod.name = "eod"
    return eod


def nan_rate(series: pd.Series) -> float:
    """Return the fraction of NaN values in *series*.  Result is in [0, 1]."""
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())
