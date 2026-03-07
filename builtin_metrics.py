"""Built-in metrics reproduced from the provided repository and paper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metric_registry import MetricSpec, register_metric, safe_divide


def _i_total(df: pd.DataFrame) -> pd.Series:
    return df["i_tp"] + df["i_fp"] + df["i_tn"] + df["i_fn"]


def _j_total(df: pd.DataFrame) -> pd.Series:
    return df["j_tp"] + df["j_fp"] + df["j_tn"] + df["j_fn"]


def _total(df: pd.DataFrame) -> pd.Series:
    return _i_total(df) + _j_total(df)


def _positive_total(df: pd.DataFrame) -> pd.Series:
    return df["i_tp"] + df["i_fn"] + df["j_tp"] + df["j_fn"]


def _negative_total(df: pd.DataFrame) -> pd.Series:
    return df["i_fp"] + df["i_tn"] + df["j_fp"] + df["j_tn"]


def group_ratio_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_j_total(df), _total(df))


def group_ratio_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_i_total(df), _total(df))


def imbalance_ratio(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_positive_total(df), _total(df))


def accuracy(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"] + df["i_tn"] + df["j_tp"] + df["j_tn"], _total(df))


def g_mean(df: pd.DataFrame) -> np.ndarray:
    tpr = safe_divide(df["i_tp"] + df["j_tp"], _positive_total(df))
    tnr = safe_divide(df["i_tn"] + df["j_tn"], _negative_total(df))
    return np.sqrt(tpr * tnr)


def tpr_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"], df["i_tp"] + df["i_fn"])


def tpr_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"], df["j_tp"] + df["j_fn"])


def fpr_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_fp"], df["i_fp"] + df["i_tn"])


def fpr_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_fp"], df["j_fp"] + df["j_tn"])


def ppv_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"], df["i_tp"] + df["i_fp"])


def ppv_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"], df["j_tp"] + df["j_fp"])


def npv_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tn"], df["i_tn"] + df["i_fn"])


def npv_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tn"], df["j_tn"] + df["j_fn"])


def acc_equality_diff(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"] + df["j_tn"], _j_total(df)) - safe_divide(df["i_tp"] + df["i_tn"], _i_total(df))


def statistical_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"] + df["j_fp"], _j_total(df)) - safe_divide(df["i_tp"] + df["i_fp"], _i_total(df))


def equal_opportunity_diff(df: pd.DataFrame) -> np.ndarray:
    return tpr_j(df) - tpr_i(df)


def predictive_equality_diff(df: pd.DataFrame) -> np.ndarray:
    return fpr_j(df) - fpr_i(df)


def positive_predictive_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return ppv_j(df) - ppv_i(df)


def negative_predictive_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return npv_j(df) - npv_i(df)


def equalized_odds_diff(df: pd.DataFrame) -> np.ndarray:
    """Equalized Odds Difference: max of |TPR gap| and |FPR gap| between groups."""
    tpr_gap = np.abs(np.asarray(tpr_j(df), dtype=np.float64) - np.asarray(tpr_i(df), dtype=np.float64))
    fpr_gap = np.abs(np.asarray(fpr_j(df), dtype=np.float64) - np.asarray(fpr_i(df), dtype=np.float64))
    result = np.full(len(df), np.nan, dtype=np.float64)
    both_defined = ~(np.isnan(tpr_gap) | np.isnan(fpr_gap))
    result[both_defined] = np.maximum(tpr_gap[both_defined], fpr_gap[both_defined])
    return result


def _odds_ratio_to_q(or_val: np.ndarray) -> np.ndarray:
    """Map an odds-ratio array to Q = (OR-1)/(OR+1)."""
    denom = or_val + 1.0
    out = np.full_like(or_val, np.nan, dtype=np.float64)
    np.divide(or_val - 1.0, denom, out=out, where=denom != 0)
    return out


def cqa_q_association(df: pd.DataFrame, smoothing: bool = True) -> np.ndarray:
    """Conditional Q Association (CQA-Q): Yule-Q RMS within each Y=y stratum.

    With smoothing=True (default, kappa=0.5) uses Haldane-Anscombe correction
    applied directly to the four cells — never pre-checks for empty strata.
    When all four cells of a stratum are zero and kappa>0, OR_y = 1 and Q_y = 0,
    so the metric remains fully defined everywhere except when kappa=0 and a
    zero denominator arises.

    2×2 table of (ŷ, S) within each Y=y stratum:
      Y=1:  a1=i_tp  b1=j_tp  c1=i_fn  d1=j_fn
      Y=0:  a0=i_fp  b0=j_fp  c0=i_tn  d0=j_tn
    """
    kappa = 0.5 if smoothing else 0.0

    a1 = np.asarray(df["i_tp"], dtype=np.float64)
    b1 = np.asarray(df["j_tp"], dtype=np.float64)
    c1 = np.asarray(df["i_fn"], dtype=np.float64)
    d1 = np.asarray(df["j_fn"], dtype=np.float64)

    a0 = np.asarray(df["i_fp"], dtype=np.float64)
    b0 = np.asarray(df["j_fp"], dtype=np.float64)
    c0 = np.asarray(df["i_tn"], dtype=np.float64)
    d0 = np.asarray(df["j_tn"], dtype=np.float64)

    # Compute OR directly from smoothed cells — no empty-stratum pre-check.
    # When kappa > 0, denominator is always > 0 so safe_divide never fires.
    # When kappa = 0, safe_divide returns NaN on zero denominators (correct).
    or1 = safe_divide((a1 + kappa) * (d1 + kappa), (b1 + kappa) * (c1 + kappa))
    or0 = safe_divide((a0 + kappa) * (d0 + kappa), (b0 + kappa) * (c0 + kappa))

    q1 = _odds_ratio_to_q(or1)
    q0 = _odds_ratio_to_q(or0)

    result = np.full(len(df), np.nan, dtype=np.float64)
    valid = ~(np.isnan(q0) | np.isnan(q1))
    result[valid] = np.sqrt((q0[valid] ** 2 + q1[valid] ** 2) / 2.0)
    return result


def fairness_phi(df: pd.DataFrame) -> np.ndarray:
    """Φ_fair — Fairness Phi (point-biserial φ between ŷ and S).

    2×2 joint table of (ŷ, S):
      n11 = i_tp + i_fp  (ŷ=1, S=1)
      n10 = j_tp + j_fp  (ŷ=1, S=0)
      n01 = i_tn + i_fn  (ŷ=0, S=1)
      n00 = j_tn + j_fn  (ŷ=0, S=0)
    """
    n11 = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n10 = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n01 = np.asarray(df["i_tn"] + df["i_fn"], dtype=np.float64)
    n00 = np.asarray(df["j_tn"] + df["j_fn"], dtype=np.float64)

    n1_dot = n11 + n10   # P(ŷ=1) * N
    n0_dot = n01 + n00   # P(ŷ=0) * N
    n_dot1 = n11 + n01   # P(S=1) * N
    n_dot0 = n10 + n00   # P(S=0) * N

    numerator = n11 * n00 - n10 * n01
    denominator_sq = n1_dot * n0_dot * n_dot1 * n_dot0
    denominator = np.sqrt(np.where(denominator_sq > 0, denominator_sq, np.nan))
    return safe_divide(numerator, denominator)


def register_builtin_metrics() -> None:
    specs = [
        MetricSpec(
            key="group_ratio_j",
            label="Group Ratio (j / total)",
            category="ratio",
            description="Group ratio following the provided repository implementation: j-group count divided by total count.",
            formula="(j_tp + j_fp + j_tn + j_fn) / total",
            compute=group_ratio_j,
        ),
        MetricSpec(
            key="group_ratio_i",
            label="Complementary Group Ratio (i / total)",
            category="ratio",
            description="Complementary group ratio: i-group count divided by total count.",
            formula="(i_tp + i_fp + i_tn + i_fn) / total",
            compute=group_ratio_i,
        ),
        MetricSpec(
            key="imbalance_ratio",
            label="Imbalance Ratio",
            category="ratio",
            description="Positive-class proportion in the full dataset.",
            formula="(i_tp + i_fn + j_tp + j_fn) / total",
            compute=imbalance_ratio,
        ),
        MetricSpec(
            key="accuracy",
            label="Accuracy",
            category="performance",
            description="Overall accuracy over the full confusion matrix.",
            formula="(i_tp + i_tn + j_tp + j_tn) / total",
            compute=accuracy,
        ),
        MetricSpec(
            key="g_mean",
            label="G-mean",
            category="performance",
            description="Geometric mean of the overall true positive rate and true negative rate.",
            formula="sqrt(TPR * TNR)",
            compute=g_mean,
        ),
        MetricSpec(
            key="tpr_i",
            label="TPR (i)",
            category="component",
            description="True positive rate for the i-group.",
            formula="i_tp / (i_tp + i_fn)",
            compute=tpr_i,
        ),
        MetricSpec(
            key="tpr_j",
            label="TPR (j)",
            category="component",
            description="True positive rate for the j-group.",
            formula="j_tp / (j_tp + j_fn)",
            compute=tpr_j,
        ),
        MetricSpec(
            key="fpr_i",
            label="FPR (i)",
            category="component",
            description="False positive rate for the i-group.",
            formula="i_fp / (i_fp + i_tn)",
            compute=fpr_i,
        ),
        MetricSpec(
            key="fpr_j",
            label="FPR (j)",
            category="component",
            description="False positive rate for the j-group.",
            formula="j_fp / (j_fp + j_tn)",
            compute=fpr_j,
        ),
        MetricSpec(
            key="ppv_i",
            label="PPV (i)",
            category="component",
            description="Positive predictive value for the i-group.",
            formula="i_tp / (i_tp + i_fp)",
            compute=ppv_i,
        ),
        MetricSpec(
            key="ppv_j",
            label="PPV (j)",
            category="component",
            description="Positive predictive value for the j-group.",
            formula="j_tp / (j_tp + j_fp)",
            compute=ppv_j,
        ),
        MetricSpec(
            key="npv_i",
            label="NPV (i)",
            category="component",
            description="Negative predictive value for the i-group.",
            formula="i_tn / (i_tn + i_fn)",
            compute=npv_i,
        ),
        MetricSpec(
            key="npv_j",
            label="NPV (j)",
            category="component",
            description="Negative predictive value for the j-group.",
            formula="j_tn / (j_tn + j_fn)",
            compute=npv_j,
        ),
        MetricSpec(
            key="accuracy_equality_difference",
            label="Accuracy Equality Difference",
            category="fairness",
            description="Difference in per-group accuracy using the repository sign convention j - i.",
            formula="(j_tp + j_tn)/j_total - (i_tp + i_tn)/i_total",
            compute=acc_equality_diff,
        ),
        MetricSpec(
            key="statistical_parity_difference",
            label="Statistical Parity Difference",
            category="fairness",
            description="Difference in predicted-positive rate using the repository sign convention j - i.",
            formula="(j_tp + j_fp)/j_total - (i_tp + i_fp)/i_total",
            compute=statistical_parity_diff,
        ),
        MetricSpec(
            key="equal_opportunity_difference",
            label="Equal Opportunity Difference",
            category="fairness",
            description="Difference in true positive rate between groups, j - i.",
            formula="j_tpr - i_tpr",
            compute=equal_opportunity_diff,
        ),
        MetricSpec(
            key="equalized_odds_difference",
            label="Equalized Odds Difference",
            category="fairness",
            description="Max of |TPR gap| and |FPR gap| between groups. Zero iff equalized odds holds.",
            formula="max(|j_tpr - i_tpr|, |j_fpr - i_fpr|)",
            compute=equalized_odds_diff,
        ),
        MetricSpec(
            key="predictive_equality_difference",
            label="Predictive Equality Difference",
            category="fairness",
            description="Difference in false positive rate between groups, j - i.",
            formula="j_fpr - i_fpr",
            compute=predictive_equality_diff,
        ),
        MetricSpec(
            key="positive_predictive_parity_difference",
            label="Positive Predictive Parity Difference",
            category="fairness",
            description="Difference in positive predictive value between groups, j - i.",
            formula="j_ppv - i_ppv",
            compute=positive_predictive_parity_diff,
        ),
        MetricSpec(
            key="negative_predictive_parity_difference",
            label="Negative Predictive Parity Difference",
            category="fairness",
            description="Difference in negative predictive value between groups, j - i.",
            formula="j_npv - i_npv",
            compute=negative_predictive_parity_diff,
        ),
        MetricSpec(
            key="cqa_q_association",
            label="Conditional Q Association",
            category="fairness",
            description=(
                "Conditional Q Association (CQA-Q): RMS of Yule-Q odds-ratio statistics within each "
                "label stratum. Bounded in [0,1). Smoothed with Haldane-Anscombe +0.5 by default."
            ),
            formula="sqrt((Q0^2 + Q1^2) / 2), OR_y with Haldane-Anscombe smoothing kappa=0.5",
            compute=cqa_q_association,
        ),
        MetricSpec(
            key="fairness_phi",
            label="Phi Fair",
            category="fairness",
            description=(
                "Phi Fair (Φ_fair): point-biserial φ coefficient between prediction ŷ and protected attribute S. "
                "Zero iff ŷ ⊥ S. Range [-1, 1]; sign indicates direction of association."
            ),
            formula="(n11*n00 - n10*n01) / sqrt(n1. * n0. * n.1 * n.0)",
            compute=fairness_phi,
        ),
    ]

    for spec in specs:
        register_metric(spec)


register_builtin_metrics()


FAIRNESS_METRIC_KEYS = [
    "accuracy_equality_difference",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "predictive_equality_difference",
    "positive_predictive_parity_difference",
    "negative_predictive_parity_difference",
    "cqa_q_association",
    "fairness_phi",
]


PERFORMANCE_METRIC_KEYS = ["accuracy", "g_mean"]
