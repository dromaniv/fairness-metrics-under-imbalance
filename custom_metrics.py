"""Custom fairness metrics registered on top of the built-in set.

Add new metrics here by defining a compute function and calling register_metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metric_registry import MetricSpec, register_metric, safe_divide
from builtin_metrics import _odds_ratio_to_q


# ---------------------------------------------------------------------------
# Fairness Phi
# ---------------------------------------------------------------------------

def fairness_phi(df: pd.DataFrame) -> np.ndarray:
    n11 = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n10 = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n01 = np.asarray(df["i_tn"] + df["i_fn"], dtype=np.float64)
    n00 = np.asarray(df["j_tn"] + df["j_fn"], dtype=np.float64)
    n1d = n11 + n10
    n0d = n01 + n00
    nd1 = n11 + n01
    nd0 = n10 + n00
    denom = np.sqrt(np.where(n1d * n0d * nd1 * nd0 > 0, n1d * n0d * nd1 * nd0, np.nan))
    return safe_divide(n11 * n00 - n10 * n01, denom)


# ---------------------------------------------------------------------------
# Marginal Q Association
# ---------------------------------------------------------------------------

def marginal_q_association(df: pd.DataFrame) -> np.ndarray:
    n11 = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n10 = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n01 = np.asarray(df["i_tn"] + df["i_fn"], dtype=np.float64)
    n00 = np.asarray(df["j_tn"] + df["j_fn"], dtype=np.float64)
    ad = n11 * n00
    bc = n10 * n01
    return safe_divide(ad - bc, ad + bc)


# ---------------------------------------------------------------------------
# Conditional Q Association
# ---------------------------------------------------------------------------

def conditional_q_association(df: pd.DataFrame, smoothing: bool = True) -> np.ndarray:
    kappa = 0.5 if smoothing else 0.0
    a1 = np.asarray(df["i_tp"], dtype=np.float64)
    b1 = np.asarray(df["j_tp"], dtype=np.float64)
    c1 = np.asarray(df["i_fn"], dtype=np.float64)
    d1 = np.asarray(df["j_fn"], dtype=np.float64)
    a0 = np.asarray(df["i_fp"], dtype=np.float64)
    b0 = np.asarray(df["j_fp"], dtype=np.float64)
    c0 = np.asarray(df["i_tn"], dtype=np.float64)
    d0 = np.asarray(df["j_tn"], dtype=np.float64)
    or1 = safe_divide((a1 + kappa) * (d1 + kappa), (b1 + kappa) * (c1 + kappa))
    or0 = safe_divide((a0 + kappa) * (d0 + kappa), (b0 + kappa) * (c0 + kappa))
    q1 = _odds_ratio_to_q(or1)
    q0 = _odds_ratio_to_q(or0)
    result = np.full(len(df), np.nan, dtype=np.float64)
    valid = ~(np.isnan(q0) | np.isnan(q1))
    result[valid] = np.sqrt((q0[valid] ** 2 + q1[valid] ** 2) / 2.0)
    return result


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(df: pd.DataFrame) -> np.ndarray:
    n11 = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n10 = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n01 = np.asarray(df["i_tn"] + df["i_fn"], dtype=np.float64)
    n00 = np.asarray(df["j_tn"] + df["j_fn"], dtype=np.float64)
    N = n11 + n10 + n01 + n00
    n1d = n11 + n10
    n0d = n01 + n00
    nd1 = n11 + n01
    nd0 = n10 + n00

    def _term(n_ys, n_y, n_s):
        denom = n_y * n_s
        ratio = np.where((n_ys > 0) & (denom > 0), safe_divide(N * n_ys, denom), np.nan)
        log_r = np.where(np.isfinite(ratio) & (ratio > 0), np.log(ratio), 0.0)
        return np.where(n_ys > 0, safe_divide(n_ys, N) * log_r, 0.0)

    result = _term(n11, n1d, nd1) + _term(n10, n1d, nd0) + _term(n01, n0d, nd1) + _term(n00, n0d, nd0)
    return np.where(N > 0, result, np.nan).astype(np.float64)


# ---------------------------------------------------------------------------
# Normalized Mutual Information
# ---------------------------------------------------------------------------

def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = (p >= 0) & (p <= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            (p > 0) & (p < 1),
            -p * np.log(np.where(p > 0, p, 1.0)) - (1 - p) * np.log(np.where(1 - p > 0, 1 - p, 1.0)),
            0.0,
        )
    out[valid] = h[valid]
    return out


def normalized_mutual_information(df: pd.DataFrame) -> np.ndarray:
    n11 = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n10 = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n01 = np.asarray(df["i_tn"] + df["i_fn"], dtype=np.float64)
    n00 = np.asarray(df["j_tn"] + df["j_fn"], dtype=np.float64)
    N = n11 + n10 + n01 + n00
    h_yhat = _binary_entropy(safe_divide(n11 + n10, N))
    h_s    = _binary_entropy(safe_divide(n11 + n01, N))
    return safe_divide(mutual_information(df), np.sqrt(h_yhat * h_s))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_metric(MetricSpec(
    key="fairness_phi",
    label="Fairness Phi",
    category="fairness",
    sort_order=100,
    description=(
        "Fairness Phi (Φ): point-biserial φ coefficient between ŷ and S. "
        "Zero iff ŷ ⊥ S. Range [−1, 1]; sign indicates direction of association."
    ),
    formula=r"\Phi = \frac{n_{11}n_{00}-n_{10}n_{01}}{\sqrt{n_{1\cdot}n_{0\cdot}n_{\cdot1}n_{\cdot0}}}",
    compute=fairness_phi,
))

register_metric(MetricSpec(
    key="marginal_q_association",
    label="Marginal Q Association",
    category="fairness",
    sort_order=101,
    description=(
        "Yule-Q on the marginal 2×2 table of (Ŷ, S). "
        "Zero iff Ŷ ⊥ S. Bounded in [−1, 1]; sign indicates direction of disparity."
    ),
    formula=r"Q = \frac{n_{11}n_{00}-n_{10}n_{01}}{n_{11}n_{00}+n_{10}n_{01}}",
    compute=marginal_q_association,
))

register_metric(MetricSpec(
    key="conditional_q_association",
    label="Conditional Q Association",
    category="fairness",
    sort_order=102,
    description=(
        "CQA: RMS of Yule-Q within each label stratum Y=y. "
        "Bounded in [0, 1). Uses Haldane-Anscombe +0.5 smoothing by default."
    ),
    formula=r"\mathrm{CQA} = \sqrt{\frac{Q_0^2+Q_1^2}{2}},\quad Q_y=\frac{\mathrm{OR}_y-1}{\mathrm{OR}_y+1}",
    compute=conditional_q_association,
))

register_metric(MetricSpec(
    key="mutual_information",
    label="Mutual Information",
    category="fairness",
    sort_order=103,
    description=(
        "Discrete MI(Ŷ; S) on the 2×2 table. "
        "Zero iff Ŷ ⊥ S. Non-negative, unsigned."
    ),
    formula=r"\mathrm{MI}=\sum_{\hat y,s}\frac{n_{\hat y s}}{N}\log\frac{N\,n_{\hat y s}}{n_{\hat y\cdot}n_{\cdot s}}",
    compute=mutual_information,
))

register_metric(MetricSpec(
    key="normalized_mutual_information",
    label="Normalized Mutual Information",
    category="fairness",
    sort_order=104,
    description=(
        "NMI = MI / sqrt(H(Ŷ)·H(S)). Scales MI to roughly [0, 1]. "
        "NaN when either marginal entropy is zero."
    ),
    formula=r"\mathrm{NMI}=\frac{\mathrm{MI}(\hat Y;S)}{\sqrt{H(\hat Y)\,H(S)}}",
    compute=normalized_mutual_information,
))
