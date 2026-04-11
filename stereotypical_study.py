"""Analysis helpers for the stereotypical bias study.

SR_p = fraction of positive predictions assigned to group j (positive stereotypical ratio).
SR_n = fraction of negative predictions assigned to group j (negative stereotypical ratio).
SR_c = √(SR_p × SR_n) — geometric mean (combined stereotypical ratio).
SR = GR_j is the proportional reference for all three.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from synthetic_analysis import ensure_metric_columns

SR_COLUMNS = {
    "sr_p": "stereotypical_ratio",
    "sr_n": "stereotypical_ratio_negative",
    "sr_c": "stereotypical_ratio_combined",
}

SR_LABELS = {
    "stereotypical_ratio": "SR_p (positive)",
    "stereotypical_ratio_negative": "SR_n (negative)",
    "stereotypical_ratio_combined": "SR_c (combined)",
}


def _ensure_sr_cols(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Ensure SR columns are present, computing via the registry if missing."""
    return ensure_metric_columns(df, list(cols))


def _filter_by_fixed(
    df: pd.DataFrame,
    ir_value: float | None,
    gr_value: float | None,
    atol: float,
) -> pd.DataFrame:
    if ir_value is not None:
        mask = np.isclose(df["imbalance_ratio"].to_numpy(np.float64), ir_value, atol=atol, rtol=0)
        df = df[mask].reset_index(drop=True)
    if gr_value is not None:
        mask = np.isclose(df["group_ratio_j"].to_numpy(np.float64), gr_value, atol=atol, rtol=0)
        df = df[mask].reset_index(drop=True)
    return df


def metric_means_by_sr(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    ir_value: float | None = None,
    gr_value: float | None = None,
    atol: float = 0.01,
    absolute: bool = False,
) -> pd.DataFrame:
    """Per-SR-value mean and std for each metric."""
    work = _ensure_sr_cols(df, sr_col)
    work = _filter_by_fixed(work, ir_value, gr_value, atol)
    work = ensure_metric_columns(work, metric_keys)

    rows: list[dict] = []
    for sr_val, group in work.groupby(sr_col, sort=True):
        row: dict = {"sr": float(sr_val)}
        for key in metric_keys:
            vals = group[key].abs() if absolute else group[key]
            valid = vals.dropna()
            row[key] = float(valid.mean()) if len(valid) > 0 else np.nan
            row[f"{key}_std"] = float(valid.std()) if len(valid) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def metric_means_by_sr_multi_ir(
    df: pd.DataFrame,
    metric_key: str,
    ir_values: list[float],
    *,
    sr_col: str = "stereotypical_ratio",
    gr_value: float | None = None,
    atol: float = 0.01,
    absolute: bool = False,
) -> pd.DataFrame:
    """Per-(SR, IR) mean and std for a single metric at a fixed GR.

    Columns: ``sr``, ``ir_value``, ``{metric_key}``, ``{metric_key}_std``.
    """
    rows: list[dict] = []
    for ir_val in ir_values:
        slice_df = metric_means_by_sr(
            df, [metric_key],
            sr_col=sr_col, ir_value=ir_val, gr_value=gr_value, atol=atol, absolute=absolute,
        )
        for _, row in slice_df.iterrows():
            rows.append({
                "sr": row["sr"],
                "ir_value": float(ir_val),
                metric_key: row.get(metric_key, np.nan),
                f"{metric_key}_std": row.get(f"{metric_key}_std", np.nan),
            })
    return pd.DataFrame(rows)


def compute_sr_sensitivity(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    ir_value: float | None = None,
    gr_value: float | None = None,
    atol: float = 0.01,
) -> pd.DataFrame:
    """Pooled Spearman ρ, OLS slope, R², and NaN fraction of each metric vs SR.

    Note: this pools all rows, so IR and GR act as confounders.
    Use ``compute_sr_sensitivity_stratified`` to control for them.

    Columns: ``metric``, ``spearman_r``, ``spearman_p``, ``slope``,
    ``r_squared``, ``nan_fraction``, ``n_valid``.
    """
    work = _ensure_sr_cols(df, sr_col)
    work = _filter_by_fixed(work, ir_value, gr_value, atol)
    work = ensure_metric_columns(work, metric_keys)

    sr = work[sr_col].to_numpy(np.float64)
    rows: list[dict] = []

    for key in metric_keys:
        vals = work[key].to_numpy(np.float64)
        nan_frac = float(np.isnan(vals).mean())
        valid = ~(np.isnan(sr) | np.isnan(vals))
        n_valid = int(valid.sum())

        if n_valid < 5:
            rows.append({
                "metric": key,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "slope": np.nan, "r_squared": np.nan,
                "nan_fraction": nan_frac, "n_valid": n_valid,
            })
            continue

        sr_v, vals_v = sr[valid], vals[valid]
        rho, p_rho = scipy_stats.spearmanr(sr_v, vals_v)
        slope, _intercept, r_val, _p, _se = scipy_stats.linregress(sr_v, vals_v)
        rows.append({
            "metric": key,
            "spearman_r": float(rho), "spearman_p": float(p_rho),
            "slope": float(slope), "r_squared": float(r_val ** 2),
            "nan_fraction": nan_frac, "n_valid": n_valid,
        })

    return pd.DataFrame(rows)


def compute_sr_sensitivity_stratified(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    strata_cols: tuple[str, ...] = ("imbalance_ratio", "group_ratio_j"),
    min_stratum_n: int = 5,
) -> pd.DataFrame:
    """Within-stratum Spearman ρ combined via Fisher Z transformation.

    Groups by *strata_cols* (default: IR × GR), computes Spearman ρ(metric, sr_col)
    within each stratum, then combines using weighted Fisher Z (weight = n − 3).
    This controls for IR and GR as confounders and avoids Simpson's paradox.

    Columns: ``metric``, ``spearman_r``, ``spearman_p``, ``nan_fraction``,
    ``n_strata``, ``n_valid``.
    """
    work = _ensure_sr_cols(df, sr_col)
    work = ensure_metric_columns(work, metric_keys)

    strata = list(work.groupby(list(strata_cols), sort=False))
    rows: list[dict] = []

    for key in metric_keys:
        z_vals: list[float] = []
        weights: list[float] = []
        n_total_valid = 0
        nan_fracs: list[float] = []

        for _, stratum in strata:
            sr_s = stratum[sr_col].to_numpy(np.float64)
            m_s = stratum[key].to_numpy(np.float64)
            valid = np.isfinite(sr_s) & np.isfinite(m_s)
            n_v = int(valid.sum())
            nan_fracs.append(float(np.isnan(m_s).mean()))
            if n_v < min_stratum_n:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy_stats.ConstantInputWarning)
                rho, _ = scipy_stats.spearmanr(sr_s[valid], m_s[valid])
            if not np.isfinite(rho):
                continue
            rho_c = float(np.clip(rho, -0.9999, 0.9999))
            z_vals.append(np.arctanh(rho_c))
            weights.append(float(n_v - 3))
            n_total_valid += n_v

        nan_frac = float(np.mean(nan_fracs)) if nan_fracs else np.nan
        n_strata = len(z_vals)

        if n_strata == 0:
            rows.append({
                "metric": key,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "nan_fraction": nan_frac, "n_strata": 0, "n_valid": 0,
            })
            continue

        total_w = sum(weights)
        z_bar = sum(w * z for w, z in zip(weights, z_vals)) / total_w
        combined_rho = float(np.tanh(z_bar))

        # Approximate p-value via z-test on the combined Fisher Z
        se = 1.0 / np.sqrt(total_w)
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z_bar) / se))) if se > 0 else np.nan

        rows.append({
            "metric": key,
            "spearman_r": combined_rho,
            "spearman_p": p_val,
            "nan_fraction": nan_frac,
            "n_strata": n_strata,
            "n_valid": n_total_valid,
        })

    return pd.DataFrame(rows)


def proportional_sr_slice(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Rows where SR ≈ GR_j (proportional prediction) within *tolerance*."""
    work = _ensure_sr_cols(df, sr_col)
    sr = work[sr_col].to_numpy(np.float64)
    gr_j = work["group_ratio_j"].to_numpy(np.float64)
    mask = np.abs(sr - gr_j) <= tolerance
    return ensure_metric_columns(work[mask].reset_index(drop=True), metric_keys)



