"""Feasible-range bounds for signed difference-based fairness metrics.

For each metric M = f(j) - f(i), these functions compute the minimum and maximum
value M can take given the fixed marginals of the confusion matrix row. The free
variable in every case is tp within each group; all denominators are fixed.

    tp_lo(K, N) = max(0, K - N)   lower Fréchet bound on tp
    tp_hi(P, K) = min(P, K)       upper Fréchet bound on tp

All functions operate vectorised over numpy arrays / pd.Series.
Returns (m_min, m_max) as a pair of np.ndarray; NaN where undefined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from metric_registry import safe_divide


@dataclass(frozen=True)
class Bounds:
    m_min: np.ndarray
    m_max: np.ndarray


# ---------------------------------------------------------------------------
# Fréchet helpers
# ---------------------------------------------------------------------------

def _tp_lo(K: np.ndarray, N: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, K - N)


def _tp_hi(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.minimum(P, K)


def _tn_from_tp(tp: np.ndarray, N: np.ndarray, K: np.ndarray) -> np.ndarray:
    """tn = N - fp = N - (K - tp) = N - K + tp"""
    return N - K + tp


def _marginals(df: pd.DataFrame) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Return (P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j) as float64 arrays."""
    P_i = np.asarray(df["i_tp"] + df["i_fn"], dtype=np.float64)
    N_i = np.asarray(df["i_fp"] + df["i_tn"], dtype=np.float64)
    K_i = np.asarray(df["i_tp"] + df["i_fp"], dtype=np.float64)
    n_i = P_i + N_i

    P_j = np.asarray(df["j_tp"] + df["j_fn"], dtype=np.float64)
    N_j = np.asarray(df["j_fp"] + df["j_tn"], dtype=np.float64)
    K_j = np.asarray(df["j_tp"] + df["j_fp"], dtype=np.float64)
    n_j = P_j + N_j

    return P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j


# ---------------------------------------------------------------------------
# SPD bounds
# ---------------------------------------------------------------------------

def spd_bounds(df: pd.DataFrame) -> Bounds:
    """SPD bounds fixing n_i, n_j and total K = K_i + K_j.

    Optimize over how many of the K total predicted positives go to group j:
        k_j in [max(0, K - n_i), min(K, n_j)]
    SPD(k_j) = k_j/n_j - (K - k_j)/n_i
    """
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)
    K = K_i + K_j
    kj_max = np.minimum(K, n_j)
    kj_min = np.maximum(0.0, K - n_i)
    m_max = safe_divide(kj_max, n_j) - safe_divide(K - kj_max, n_i)
    m_min = safe_divide(kj_min, n_j) - safe_divide(K - kj_min, n_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# EOD bounds  (TPR difference)
# ---------------------------------------------------------------------------

def eod_bounds(df: pd.DataFrame) -> Bounds:
    """EOD = tp_j/P_j - tp_i/P_i."""
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)

    tphi_j = _tp_hi(P_j, K_j)
    tplo_j = _tp_lo(K_j, N_j)
    tphi_i = _tp_hi(P_i, K_i)
    tplo_i = _tp_lo(K_i, N_i)

    m_max = safe_divide(tphi_j, P_j) - safe_divide(tplo_i, P_i)
    m_min = safe_divide(tplo_j, P_j) - safe_divide(tphi_i, P_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# FPRD bounds  (FPR difference)
# ---------------------------------------------------------------------------

def fprd_bounds(df: pd.DataFrame) -> Bounds:
    """FPRD = fp_j/N_j - fp_i/N_i.  fp = K - tp, so fp_lo = max(0, K-P), fp_hi = min(K, N)."""
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)

    fphi_j = np.minimum(K_j, N_j)          # = K_j - tp_lo_j
    fplo_j = np.maximum(0.0, K_j - P_j)    # = K_j - tp_hi_j
    fphi_i = np.minimum(K_i, N_i)
    fplo_i = np.maximum(0.0, K_i - P_i)

    m_max = safe_divide(fphi_j, N_j) - safe_divide(fplo_i, N_i)
    m_min = safe_divide(fplo_j, N_j) - safe_divide(fphi_i, N_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# ACCD bounds  (accuracy equality difference)
# ---------------------------------------------------------------------------

def accd_bounds(df: pd.DataFrame) -> Bounds:
    """ACCD = (tp+tn)/n per group.  tp+tn = 2*tp + N - K (linear in tp)."""
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)

    tphi_j = _tp_hi(P_j, K_j)
    tplo_j = _tp_lo(K_j, N_j)
    tphi_i = _tp_hi(P_i, K_i)
    tplo_i = _tp_lo(K_i, N_i)

    def _acc(tp: np.ndarray, N: np.ndarray, K: np.ndarray, n: np.ndarray) -> np.ndarray:
        return safe_divide(2.0 * tp + N - K, n)

    m_max = _acc(tphi_j, N_j, K_j, n_j) - _acc(tplo_i, N_i, K_i, n_i)
    m_min = _acc(tplo_j, N_j, K_j, n_j) - _acc(tphi_i, N_i, K_i, n_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# PPVD bounds  (PPV difference)
# ---------------------------------------------------------------------------

def ppvd_bounds(df: pd.DataFrame) -> Bounds:
    """PPVD = tp_j/K_j - tp_i/K_i.  K is fixed → linear in tp."""
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)

    tphi_j = _tp_hi(P_j, K_j)
    tplo_j = _tp_lo(K_j, N_j)
    tphi_i = _tp_hi(P_i, K_i)
    tplo_i = _tp_lo(K_i, N_i)

    m_max = safe_divide(tphi_j, K_j) - safe_divide(tplo_i, K_i)
    m_min = safe_divide(tplo_j, K_j) - safe_divide(tphi_i, K_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# NPVD bounds  (NPV difference)
# ---------------------------------------------------------------------------

def npvd_bounds(df: pd.DataFrame) -> Bounds:
    """NPVD = tn_j/R_j - tn_i/R_i where R = n - K (predicted negatives).
    tn = N - K + tp (linear in tp).
    """
    P_i, N_i, K_i, n_i, P_j, N_j, K_j, n_j = _marginals(df)

    R_i = n_i - K_i  # predicted negatives in group i
    R_j = n_j - K_j

    tphi_j = _tp_hi(P_j, K_j)
    tplo_j = _tp_lo(K_j, N_j)
    tphi_i = _tp_hi(P_i, K_i)
    tplo_i = _tp_lo(K_i, N_i)

    tnhi_j = _tn_from_tp(tphi_j, N_j, K_j)
    tnlo_j = _tn_from_tp(tplo_j, N_j, K_j)
    tnhi_i = _tn_from_tp(tphi_i, N_i, K_i)
    tnlo_i = _tn_from_tp(tplo_i, N_i, K_i)

    m_max = safe_divide(tnhi_j, R_j) - safe_divide(tnlo_i, R_i)
    m_min = safe_divide(tnlo_j, R_j) - safe_divide(tnhi_i, R_i)
    return Bounds(m_min=m_min, m_max=m_max)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_BOUNDS_REGISTRY: dict[str, Callable[[pd.DataFrame], Bounds]] = {
    "statistical_parity_difference":        spd_bounds,
    "equal_opportunity_difference":         eod_bounds,
    "predictive_equality_difference":       fprd_bounds,
    "accuracy_equality_difference":         accd_bounds,
    "positive_predictive_parity_difference": ppvd_bounds,
    "negative_predictive_parity_difference": npvd_bounds,
}

BOUNDS_SUPPORTED_KEYS: frozenset[str] = frozenset(_BOUNDS_REGISTRY)


def compute_bounds(metric_key: str, df: pd.DataFrame) -> Bounds:
    """Return feasible (m_min, m_max) for *metric_key* over all rows of *df*.

    Raises KeyError for metrics with no bounds implementation.
    """
    if metric_key not in _BOUNDS_REGISTRY:
        raise KeyError(
            f"No bounds implementation for metric '{metric_key}'. "
            f"Supported: {sorted(BOUNDS_SUPPORTED_KEYS)}"
        )
    return _BOUNDS_REGISTRY[metric_key](df)


# ---------------------------------------------------------------------------
# FRN vectorised helper
# ---------------------------------------------------------------------------

def frn(raw: np.ndarray, m_min: np.ndarray, m_max: np.ndarray) -> np.ndarray:
    """Feasible-Range Normalization applied element-wise.

    frn(M) = 0          if M == 0
           = M / M_max  if M > 0  (M_max > 0 required, else 0)
           = M / |M_min| if M < 0  (M_min < 0 required, else 0)

    Returns NaN where raw, m_min, or m_max is NaN.
    """
    raw  = np.asarray(raw,   dtype=np.float64)
    mmin = np.asarray(m_min, dtype=np.float64)
    mmax = np.asarray(m_max, dtype=np.float64)

    out = np.full_like(raw, np.nan, dtype=np.float64)

    pos  = raw > 0
    neg  = raw < 0
    zero = raw == 0.0

    out[zero] = 0.0
    np.divide(raw, mmax,         out=out, where=pos & (mmax > 0))
    np.divide(raw, np.abs(mmin), out=out, where=neg & (mmin < 0))

    # edge: M>0 but M_max<=0, or M<0 but M_min>=0 → infeasible, return 0
    out[pos & (mmax <= 0)] = 0.0
    out[neg & (mmin >= 0)] = 0.0

    return out


