"""Analysis helpers for the synthetic confusion-matrix study."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from metric_registry import compute_metric


def resolve_ratio_column(ratio_type: str, group_ratio_basis: str = "j") -> str:
    ratio_type = ratio_type.lower()
    if ratio_type == "ir":
        return "imbalance_ratio"
    if ratio_type == "gr":
        basis = group_ratio_basis.lower()
        if basis not in {"i", "j"}:
            raise ValueError("group_ratio_basis must be 'i' or 'j'")
        return f"group_ratio_{basis}"
    raise ValueError("ratio_type must be 'ir' or 'gr'")


def ensure_metric_column(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    if metric_key in df.columns:
        return df
    out = df.copy()
    out[metric_key] = compute_metric(out, metric_key)
    return out


def ensure_metric_columns(df: pd.DataFrame, metric_keys: Iterable[str]) -> pd.DataFrame:
    out = df
    for key in metric_keys:
        out = ensure_metric_column(out, key)
    return out


def ratio_mask(series: pd.Series, value: float, atol: float = 1e-9) -> np.ndarray:
    return np.isclose(series.to_numpy(dtype=np.float64), float(value), atol=atol, rtol=0.0)



def select_histogram_slice(
    df: pd.DataFrame,
    metric_key: str,
    gr_values: Iterable[float],
    ir_values: Iterable[float],
    *,
    group_ratio_basis: str = "j",
) -> pd.DataFrame:
    out = ensure_metric_column(df, metric_key)
    gr_col = resolve_ratio_column("gr", group_ratio_basis)
    ir_col = resolve_ratio_column("ir", group_ratio_basis)
    gr_mask = np.zeros(len(out), dtype=bool)
    ir_mask = np.zeros(len(out), dtype=bool)
    for value in gr_values:
        gr_mask |= ratio_mask(out[gr_col], value)
    for value in ir_values:
        ir_mask |= ratio_mask(out[ir_col], value)
    return out.loc[gr_mask & ir_mask].copy()



def probability_of_perfect_fairness(
    df: pd.DataFrame,
    metric_keys: Iterable[str],
    ratio_type: str,
    *,
    epsilon: float = 0.0,
    group_ratio_basis: str = "j",
) -> pd.DataFrame:
    ratio_col = resolve_ratio_column(ratio_type, group_ratio_basis)
    out = ensure_metric_columns(df, metric_keys)

    rows = []
    grouped = out.groupby(ratio_col, sort=True)
    for ratio_value, group in grouped:
        row = {ratio_type: ratio_value}
        for key in metric_keys:
            values = group[key]
            if values.isna().all():
                row[key] = np.nan
            else:
                if epsilon == 0:
                    row[key] = float((values == 0).sum() / len(values))
                else:
                    row[key] = float((values.abs() < epsilon).sum() / len(values))
        rows.append(row)
    return pd.DataFrame(rows)



def probability_of_nan(
    df: pd.DataFrame,
    metric_keys: Iterable[str],
    ratio_type: str,
    *,
    group_ratio_basis: str = "j",
) -> pd.DataFrame:
    ratio_col = resolve_ratio_column(ratio_type, group_ratio_basis)
    out = ensure_metric_columns(df, metric_keys)

    rows = []
    grouped = out.groupby(ratio_col, sort=True)
    for ratio_value, group in grouped:
        row = {ratio_type: ratio_value}
        for key in metric_keys:
            row[key] = float(group[key].isna().mean())
        rows.append(row)
    return pd.DataFrame(rows)



def value_grid_for_heatmap(
    df: pd.DataFrame,
    fairness_key: str,
    performance_key: str,
    *,
    fairness_range: tuple[float, float] = (-1.0, 1.0),
    performance_range: tuple[float, float] = (0.0, 1.0),
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = ensure_metric_columns(df, [fairness_key, performance_key])
    subset = out[[fairness_key, performance_key]].dropna()
    if subset.empty:
        raise ValueError("No finite values available for the selected metrics.")

    heat, x_edges, y_edges = np.histogram2d(
        subset[fairness_key].to_numpy(dtype=np.float64),
        subset[performance_key].to_numpy(dtype=np.float64),
        bins=bins,
        range=[list(fairness_range), list(performance_range)],
    )
    heat = heat / heat.sum()
    return heat.T, x_edges, y_edges
