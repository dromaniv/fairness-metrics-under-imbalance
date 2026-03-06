"""
wrapper.py
----------
Two-stage wrapper for difference-based fairness metrics.

Stage 1 — Jeffreys smoothing
    Replace raw plug-in rates with Jeffreys-posterior means:

        r̃ₐ = (xₐ + ½) / (dₐ + 1)   ∈ (0, 1)  always defined

    Result: smoothed difference  r̃ⱼ − r̃ᵢ  ∈ (−1, 1).
    Use ``apply_jeffreys_df`` for this stage alone.

Stage 2 — Variance-Stabilising Transform (VST / arcsin)
    Apply the arcsine transform on top of the smoothed rates:

        W = (2/π) · (arcsin(√r̃ⱼ) − arcsin(√r̃ᵢ))  ∈ [−1, 1]

    Var(arcsin(√p̂)) ≈ 1/(4n) regardless of p → variance is stable.
    Use ``apply_wrapper_df`` for both stages combined.

Supported metrics
-----------------
  "eod"  – Equal Opportunity    x=TP,    d=TP+FN
  "fprd" – Predictive Equality  x=FP,    d=FP+TN
  "spd"  – Statistical Parity   x=TP+FP, d=n
  "ppvd" – Pos. Pred. Parity    x=TP,    d=TP+FP
  "npvd" – Neg. Pred. Parity    x=TN,    d=TN+FN
  "accd" – Accuracy Equality    x=TP+TN, d=n

Public API
----------
  apply_jeffreys_df(df, metric)  → pd.Series  (smoothed difference only)
  apply_wrapper_df(df, metric)   → pd.Series  (smoothed + VST)
  wrap(cm_i, cm_j, *, metric)   → float       (scalar, smoothed + VST)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Jeffreys prior defaults
_ALPHA = 0.5
_BETA  = 0.5

# Valid metric keys (also used for the wrapper's rate extraction)
WRAPPER_METRICS = ("eod", "fprd", "eqod", "spd", "ppvd", "npvd", "accd")


# ---------------------------------------------------------------------------
# Internal: (x, d) extraction
# ---------------------------------------------------------------------------

def _extract_counts_df(
    df: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x0, d0, x1, d1) float64 arrays for the chosen metric."""
    tp_i = df["i_tp"].values.astype(np.float64)
    fp_i = df["i_fp"].values.astype(np.float64)
    tn_i = df["i_tn"].values.astype(np.float64)
    fn_i = df["i_fn"].values.astype(np.float64)
    tp_j = df["j_tp"].values.astype(np.float64)
    fp_j = df["j_fp"].values.astype(np.float64)
    tn_j = df["j_tn"].values.astype(np.float64)
    fn_j = df["j_fn"].values.astype(np.float64)

    if metric == "eod":
        return tp_i, tp_i + fn_i, tp_j, tp_j + fn_j
    elif metric == "fprd":
        return fp_i, fp_i + tn_i, fp_j, fp_j + tn_j
    elif metric == "spd":
        n_i = tp_i + fp_i + tn_i + fn_i
        n_j = tp_j + fp_j + tn_j + fn_j
        return tp_i + fp_i, n_i, tp_j + fp_j, n_j
    elif metric == "ppvd":
        return tp_i, tp_i + fp_i, tp_j, tp_j + fp_j
    elif metric == "npvd":
        return tn_i, tn_i + fn_i, tn_j, tn_j + fn_j
    elif metric == "accd":
        n_i = tp_i + fp_i + tn_i + fn_i
        n_j = tp_j + fp_j + tn_j + fn_j
        return tp_i + tn_i, n_i, tp_j + tn_j, n_j
    else:
        raise ValueError(
            f"Unknown wrapper metric '{metric}'. "
            f"Choose from: {', '.join(WRAPPER_METRICS)}."
        )


# ---------------------------------------------------------------------------
# Core VST computation (vectorised)
# ---------------------------------------------------------------------------

def _vst(
    x0: np.ndarray, d0: np.ndarray,
    x1: np.ndarray, d1: np.ndarray,
    alpha: float = _ALPHA,
    beta: float  = _BETA,
) -> np.ndarray:
    """Jeffreys-smoothed arcsine VST over 1-D arrays. Returns values in [−1, 1]."""
    r0 = (x0 + alpha) / (d0 + alpha + beta)
    r1 = (x1 + alpha) / (d1 + alpha + beta)
    return (2.0 / math.pi) * (np.arcsin(np.sqrt(r1)) - np.arcsin(np.sqrt(r0)))


def _smoothed_diff(
    x0: np.ndarray, d0: np.ndarray,
    x1: np.ndarray, d1: np.ndarray,
    alpha: float = _ALPHA,
    beta: float  = _BETA,
) -> np.ndarray:
    """Jeffreys-smoothed difference r̃₁ − r̃₀, no arcsin.  ∈ (−1, 1)."""
    r0 = (x0 + alpha) / (d0 + alpha + beta)
    r1 = (x1 + alpha) / (d1 + alpha + beta)
    return r1 - r0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_jeffreys_df(
    df: pd.DataFrame,
    metric: str = "eod",
    *,
    alpha_prior: float = _ALPHA,
    beta_prior:  float = _BETA,
) -> pd.Series:
    """Stage 1 only: replace raw rates with Jeffreys-smoothed rates, return difference.

    r̃ₐ = (xₐ + ½) / (dₐ + 1)  — always in (0, 1), no 0/0.
    Result = r̃ⱼ − r̃ᵢ  ∈ (−1, 1).
    For eqod: max(|smoothed_eod|, |smoothed_fprd|).
    """
    if metric == "eqod":
        eod  = apply_jeffreys_df(df, "eod",  alpha_prior=alpha_prior, beta_prior=beta_prior)
        fprd = apply_jeffreys_df(df, "fprd", alpha_prior=alpha_prior, beta_prior=beta_prior)
        return eod.abs().combine(fprd.abs(), max).rename("jeffreys")
    x0, d0, x1, d1 = _extract_counts_df(df, metric)
    w = _smoothed_diff(x0, d0, x1, d1, alpha=alpha_prior, beta=beta_prior)
    return pd.Series(w, index=df.index, name="jeffreys", dtype=np.float64)


def apply_wrapper_df(
    df: pd.DataFrame,
    metric: str = "eod",
    *,
    alpha_prior: float = _ALPHA,
    beta_prior:  float = _BETA,
) -> pd.Series:
    """Stage 1 + 2: Jeffreys smoothing followed by arcsin VST.

    W = (2/π)·(arcsin(√r̃ⱼ) − arcsin(√r̃ᵢ))  ∈ [−1, 1]
    For eqod: max(|vst_eod|, |vst_fprd|).
    """
    if metric == "eqod":
        eod  = apply_wrapper_df(df, "eod",  alpha_prior=alpha_prior, beta_prior=beta_prior)
        fprd = apply_wrapper_df(df, "fprd", alpha_prior=alpha_prior, beta_prior=beta_prior)
        return eod.abs().combine(fprd.abs(), max).rename("w_vst")
    x0, d0, x1, d1 = _extract_counts_df(df, metric)
    w = _vst(x0, d0, x1, d1, alpha=alpha_prior, beta=beta_prior)
    return pd.Series(w, index=df.index, name="w_vst", dtype=np.float64)


def wrap(
    cm_i: dict[str, int],
    cm_j: dict[str, int],
    *,
    metric: str = "eod",
    alpha_prior: float = _ALPHA,
    beta_prior:  float = _BETA,
) -> float:
    """Scalar VST wrapper for a single confusion-matrix pair.

    Parameters
    ----------
    cm_i, cm_j   : Dicts with keys tp, fp, tn, fn.
    metric       : Rate-based metric key.
    alpha_prior  : Beta prior α (default ½).
    beta_prior   : Beta prior β (default ½).

    Returns
    -------
    float in [−1, 1].
    """
    def _get(cm: dict, k: str) -> float:
        return float(cm.get(k, 0))

    tp_i, fp_i = _get(cm_i, "tp"), _get(cm_i, "fp")
    tn_i, fn_i = _get(cm_i, "tn"), _get(cm_i, "fn")
    tp_j, fp_j = _get(cm_j, "tp"), _get(cm_j, "fp")
    tn_j, fn_j = _get(cm_j, "tn"), _get(cm_j, "fn")

    if metric == "eod":
        x0, d0, x1, d1 = tp_i, tp_i+fn_i, tp_j, tp_j+fn_j
    elif metric == "fprd":
        x0, d0, x1, d1 = fp_i, fp_i+tn_i, fp_j, fp_j+tn_j
    elif metric == "spd":
        x0, d0 = tp_i+fp_i, tp_i+fp_i+tn_i+fn_i
        x1, d1 = tp_j+fp_j, tp_j+fp_j+tn_j+fn_j
    elif metric == "ppvd":
        x0, d0, x1, d1 = tp_i, tp_i+fp_i, tp_j, tp_j+fp_j
    elif metric == "npvd":
        x0, d0, x1, d1 = tn_i, tn_i+fn_i, tn_j, tn_j+fn_j
    elif metric == "accd":
        x0, d0 = tp_i+tn_i, tp_i+fp_i+tn_i+fn_i
        x1, d1 = tp_j+tn_j, tp_j+fp_j+tn_j+fn_j
    else:
        raise ValueError(f"Unknown wrapper metric '{metric}'.")

    return float(_vst(
        np.array([x0]), np.array([d0]),
        np.array([x1]), np.array([d1]),
        alpha=alpha_prior, beta=beta_prior,
    )[0])
