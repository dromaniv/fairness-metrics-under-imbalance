"""
metrics.py
----------
All fairness metrics, fully vectorised over the 8-column CM-pair DataFrame
from generate_cm_pairs().

  i_tp, i_fp, i_tn, i_fn  —  protected group   (i)
  j_tp, j_fp, j_tn, j_fn  —  unprotected group (j)

Per-group rates (g ∈ {i, j}):
  TPR_g = TP_g / (TP_g + FN_g)
  FPR_g = FP_g / (FP_g + TN_g)
  PPV_g = TP_g / (TP_g + FP_g)
  NPV_g = TN_g / (TN_g + FN_g)
  SR_g  = (TP_g + FP_g) / N_g
  ACC_g = (TP_g + TN_g) / N_g

Metrics
-------
  accd   ACC_j − ACC_i                          ∈ [−1, 1]
  spd    SR_j  − SR_i                           ∈ [−1, 1]
  eod    TPR_j − TPR_i                          ∈ [−1, 1]
  fprd   FPR_j − FPR_i                          ∈ [−1, 1]
  eqod   max(|eod|, |fprd|)                     ∈ [0,  1]
  ppvd   PPV_j − PPV_i                          ∈ [−1, 1]
  npvd   NPV_j − NPV_i                          ∈ [−1, 1]
  caf_q_rms_ha      sqrt((Q0²+Q1²)/2)           ∈ [0,  1)   unsigned
  caf_q_rms_ha_abs  sign(Q0+Q1)·sqrt(…)         ∈ (−1, 1)   signed
  caf_eopp_q_ha     |Q1|                         ∈ [0,  1)

CAF stratum cell layout  (2×2 table of (Ŷ, S) within Y=y):
  y=1: a=FN_i, b=FN_j, c=TP_i, d=TP_j
  y=0: a=TN_i, b=TN_j, c=FP_i, d=FP_j

  OR_y = (a_y+κ)(d_y+κ) / ((b_y+κ)(c_y+κ))
  Q_y  = (OR_y − 1) / (OR_y + 1)

  caf_q_rms_ha     = sqrt((Q0² + Q1²) / 2)               [unsigned, per spec]
  caf_q_rms_ha_abs = sign(Q0 + Q1) · sqrt((Q0² + Q1²) / 2)  [signed]
  caf_eopp_q_ha    = |Q1|

NaN policies (classic metrics only)
  "nan"          0/0 → NaN
  "smooth_alpha" (x + α) / (d + k·α)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

NanPolicy = Literal["nan", "smooth_alpha"]
_VALID_POLICIES: tuple[str, ...] = ("nan", "smooth_alpha")

METRIC_LABELS: dict[str, str] = {
    "accd":              "Accuracy Equality",
    "spd":               "Statistical Parity",
    "eod":               "Equal Opportunity",
    "fprd":              "Predictive Equality",
    "eqod":              "Equalized Odds",
    "ppvd":              "Positive Predictive Parity",
    "npvd":              "Negative Predictive Parity",
    "caf_q_rms_ha":      "CAF-Q (unsigned)",
    "caf_q_rms_ha_abs":  "CAF-Q (signed)",
    "caf_eopp_q_ha":     "CAF Equal Opportunity",
}

METRIC_RANGE: dict[str, tuple[float | None, float | None]] = {
    "accd":             (-1.0,  1.0),
    "spd":              (-1.0,  1.0),
    "eod":              (-1.0,  1.0),
    "fprd":             (-1.0,  1.0),
    "eqod":             ( 0.0,  1.0),
    "ppvd":             (-1.0,  1.0),
    "npvd":             (-1.0,  1.0),
    "caf_q_rms_ha":     ( 0.0,  1.0),
    "caf_q_rms_ha_abs": (-1.0,  1.0),
    "caf_eopp_q_ha":    ( 0.0,  1.0),
}

PERFECT_FAIRNESS_ZERO: frozenset[str] = frozenset(METRIC_LABELS.keys())

METRIC_SLICE_DEFAULTS: dict[str, tuple[float, float]] = {k: (0.5, 0.5) for k in METRIC_LABELS}


# ── classic rate helpers ──────────────────────────────────────────────────

def _safe_div(
    num: pd.Series, den: pd.Series,
    policy: NanPolicy, alpha: float, k: int = 2,
) -> pd.Series:
    if policy == "nan":
        return num / den.where(den != 0, other=np.nan)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0 for smooth_alpha; got {alpha}")
    return (num + alpha) / (den + k * alpha)


def _tpr(tp, fn, policy, alpha):
    return _safe_div(tp, tp + fn, policy, alpha)

def _fpr(fp, tn, policy, alpha):
    return _safe_div(fp, fp + tn, policy, alpha)

def _sr(tp, fp, tn, fn, policy, alpha):
    return _safe_div(tp + fp, tp + fp + tn + fn, policy, alpha, k=4)

def _ppv(tp, fp, policy, alpha):
    return _safe_div(tp, tp + fp, policy, alpha)

def _npv(tn, fn, policy, alpha):
    return _safe_div(tn, tn + fn, policy, alpha)

def _acc(tp, fp, tn, fn, policy, alpha):
    return _safe_div(tp + tn, tp + fp + tn + fn, policy, alpha, k=4)


# ── CAF cell extraction & Yule's Q ───────────────────────────────────────

def _stratum_arrays(df: pd.DataFrame):
    """Extract float64 numpy arrays for both strata.

    y=1: (a=FN_i, b=FN_j, c=TP_i, d=TP_j)
    y=0: (a=TN_i, b=TN_j, c=FP_i, d=FP_j)

    Returns fn_i, fn_j, tp_i, tp_j, tn_i, tn_j, fp_i, fp_j.
    """
    return (
        df["i_fn"].to_numpy(np.float64), df["j_fn"].to_numpy(np.float64),
        df["i_tp"].to_numpy(np.float64), df["j_tp"].to_numpy(np.float64),
        df["i_tn"].to_numpy(np.float64), df["j_tn"].to_numpy(np.float64),
        df["i_fp"].to_numpy(np.float64), df["j_fp"].to_numpy(np.float64),
    )


def _yule_q_arrays(a, b, c, d, kappa: float = 0.5) -> np.ndarray:
    """Yule's Q via smoothed OR.

    OR_y = (a+κ)(d+κ) / ((b+κ)(c+κ))
    Q_y  = (OR_y − 1) / (OR_y + 1)
    """
    a_ = a + kappa; b_ = b + kappa; c_ = c + kappa; d_ = d + kappa
    OR = (a_ * d_) / (b_ * c_)
    return (OR - 1.0) / (OR + 1.0)


# ── backward-compat shims used by wrapper.py ─────────────────────────────

def _stratum_cells(df: pd.DataFrame):
    cells_y1 = (df["i_fn"], df["j_fn"], df["i_tp"], df["j_tp"])
    cells_y0 = (df["i_tn"], df["j_tn"], df["i_fp"], df["j_fp"])
    return cells_y1, cells_y0


def _yule_q(a, b, c, d, kappa: float = 0.5) -> pd.Series:
    idx = a.index if hasattr(a, "index") else None
    return pd.Series(
        _yule_q_arrays(
            np.asarray(a, np.float64), np.asarray(b, np.float64),
            np.asarray(c, np.float64), np.asarray(d, np.float64),
            kappa=kappa,
        ),
        index=idx,
    )


def _signed_agg_rms(u0: pd.Series, u1: pd.Series) -> pd.Series:
    v0 = u0.to_numpy(np.float64); v1 = u1.to_numpy(np.float64)
    return pd.Series(np.sign(v0 + v1) * np.sqrt((v0**2 + v1**2) / 2.0), index=u0.index)


# ── main dispatch ─────────────────────────────────────────────────────────

def compute_metric(
    df: pd.DataFrame,
    metric: str = "eod",
    policy: NanPolicy = "nan",
    alpha: float = 1.0,
    kappa: float = 0.5,
) -> pd.Series:
    """Compute a fairness metric for every row in df.

    Returns pd.Series aligned to df.index.
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(f"Unknown NaN policy '{policy}'.")

    idx = df.index
    kw  = dict(policy=policy, alpha=alpha)

    if metric == "accd":
        result = (_acc(df["j_tp"], df["j_fp"], df["j_tn"], df["j_fn"], **kw)
                - _acc(df["i_tp"], df["i_fp"], df["i_tn"], df["i_fn"], **kw))

    elif metric == "spd":
        result = (_sr(df["j_tp"], df["j_fp"], df["j_tn"], df["j_fn"], **kw)
                - _sr(df["i_tp"], df["i_fp"], df["i_tn"], df["i_fn"], **kw))

    elif metric == "eod":
        result = _tpr(df["j_tp"], df["j_fn"], **kw) - _tpr(df["i_tp"], df["i_fn"], **kw)

    elif metric == "fprd":
        result = _fpr(df["j_fp"], df["j_tn"], **kw) - _fpr(df["i_fp"], df["i_tn"], **kw)

    elif metric == "ppvd":
        result = _ppv(df["j_tp"], df["j_fp"], **kw) - _ppv(df["i_tp"], df["i_fp"], **kw)

    elif metric == "npvd":
        result = _npv(df["j_tn"], df["j_fn"], **kw) - _npv(df["i_tn"], df["i_fn"], **kw)

    elif metric == "eqod":
        tp_i = df["i_tp"].to_numpy(np.float64); fn_i = df["i_fn"].to_numpy(np.float64)
        tp_j = df["j_tp"].to_numpy(np.float64); fn_j = df["j_fn"].to_numpy(np.float64)
        fp_i = df["i_fp"].to_numpy(np.float64); tn_i = df["i_tn"].to_numpy(np.float64)
        fp_j = df["j_fp"].to_numpy(np.float64); tn_j = df["j_tn"].to_numpy(np.float64)

        if policy == "nan":
            with np.errstate(invalid="ignore", divide="ignore"):
                tpr_i = np.where(tp_i + fn_i > 0, tp_i / (tp_i + fn_i), np.nan)
                tpr_j = np.where(tp_j + fn_j > 0, tp_j / (tp_j + fn_j), np.nan)
                fpr_i = np.where(fp_i + tn_i > 0, fp_i / (fp_i + tn_i), np.nan)
                fpr_j = np.where(fp_j + tn_j > 0, fp_j / (fp_j + tn_j), np.nan)
        else:
            tpr_i = (tp_i + alpha) / (tp_i + fn_i + 2 * alpha)
            tpr_j = (tp_j + alpha) / (tp_j + fn_j + 2 * alpha)
            fpr_i = (fp_i + alpha) / (fp_i + tn_i + 2 * alpha)
            fpr_j = (fp_j + alpha) / (fp_j + tn_j + 2 * alpha)

        result = pd.Series(np.maximum(np.abs(tpr_j - tpr_i), np.abs(fpr_j - fpr_i)), index=idx)

    elif metric == "caf_q_rms_ha":
        # unsigned: sqrt((Q0² + Q1²) / 2)  ∈ [0, 1)
        fn_i, fn_j, tp_i, tp_j, tn_i, tn_j, fp_i, fp_j = _stratum_arrays(df)
        q1 = _yule_q_arrays(fn_i, fn_j, tp_i, tp_j, kappa=kappa)
        q0 = _yule_q_arrays(tn_i, tn_j, fp_i, fp_j, kappa=kappa)
        result = pd.Series(np.sqrt((q0**2 + q1**2) / 2.0), index=idx)

    elif metric == "caf_q_rms_ha_abs":
        # signed: sign(Q0+Q1) · sqrt((Q0² + Q1²) / 2)  ∈ (−1, 1)
        fn_i, fn_j, tp_i, tp_j, tn_i, tn_j, fp_i, fp_j = _stratum_arrays(df)
        q1 = _yule_q_arrays(fn_i, fn_j, tp_i, tp_j, kappa=kappa)
        q0 = _yule_q_arrays(tn_i, tn_j, fp_i, fp_j, kappa=kappa)
        result = pd.Series(np.sign(q0 + q1) * np.sqrt((q0**2 + q1**2) / 2.0), index=idx)

    elif metric == "caf_eopp_q_ha":
        fn_i, fn_j, tp_i, tp_j, *_ = _stratum_arrays(df)
        result = pd.Series(np.abs(_yule_q_arrays(fn_i, fn_j, tp_i, tp_j, kappa=kappa)), index=idx)

    elif metric in ("caf", "caf_eopp"):
        import warnings
        alias = "caf_q_rms_ha" if metric == "caf" else "caf_eopp_q_ha"
        warnings.warn(f"'{metric}' is deprecated; use '{alias}'", DeprecationWarning, stacklevel=2)
        return compute_metric(df, metric=alias, policy=policy, alpha=alpha, kappa=kappa)

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {', '.join(METRIC_LABELS)}.")

    result.name = metric
    return result


def compute_eod(df: pd.DataFrame, policy: NanPolicy = "nan", alpha: float = 1.0) -> pd.Series:
    return compute_metric(df, metric="eod", policy=policy, alpha=alpha)


def nan_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def perfect_fairness_rate(series: pd.Series, eps: float = 0.0) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid.abs() <= eps).mean())

