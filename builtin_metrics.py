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
    ]

    for spec in specs:
        register_metric(spec)


register_builtin_metrics()


FAIRNESS_METRIC_KEYS = [
    "accuracy_equality_difference",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "predictive_equality_difference",
    "positive_predictive_parity_difference",
    "negative_predictive_parity_difference",
]


PERFORMANCE_METRIC_KEYS = ["accuracy", "g_mean"]
