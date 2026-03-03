"""
plots.py
--------
Slicing helpers and Plotly figure factories for the fairness-wrapper dashboard.

All functions accept the fully annotated DataFrame produced by
``generate_cm_pairs()`` with an additional ``eod`` column attached, e.g.::

    df = generate_cm_pairs(n=24)
    df["eod"] = compute_eod(df, policy="nan")

Figure factories return ``plotly.graph_objects.Figure`` objects that can be
passed directly to ``st.plotly_chart()`` in the Streamlit app.

Binning strategy
----------------
IR and GR are continuous but limited to rational multiples of 1/n.
We bin them into ``n_bins`` equal-width buckets across [0, 1] and
summarise each bucket with mean ± std.  Buckets with fewer than
``min_count`` observations are shown as a gap in the line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_DEFAULT_BINS = 20
_MIN_COUNT = 1  # show even single-sample bins; increase to 3 for smoother lines


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------

def _filter_near(
    df: pd.DataFrame,
    col: str,
    center: float,
    tol: float,
) -> pd.DataFrame:
    """Return rows where ``|df[col] − center| ≤ tol``."""
    return df[np.abs(df[col] - center) <= tol]


def _bin_and_summarise(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = _DEFAULT_BINS,
    min_count: int = _MIN_COUNT,
) -> pd.DataFrame:
    """Bin *x_col* and compute mean, std, count of *y_col* per bin.

    Returns a DataFrame with columns: bin_mid, mean, std, count.
    Bins with < min_count observations are excluded.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])  # bin mid-points

    df = df.copy()
    df["_bin"] = pd.cut(df[x_col], bins=edges, labels=labels, include_lowest=True)

    agg = (
        df.groupby("_bin", observed=True)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"_bin": "bin_mid"})
    )
    agg["bin_mid"] = agg["bin_mid"].astype(float)
    agg = agg[agg["count"] >= min_count]
    return agg


def _nan_rate_by_bin(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "eod",
    n_bins: int = _DEFAULT_BINS,
    min_count: int = _MIN_COUNT,
) -> pd.DataFrame:
    """Compute NaN fraction of *y_col* per bin of *x_col*.

    Returns a DataFrame with columns: bin_mid, nan_rate, count.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])

    df = df.copy()
    df["_bin"] = pd.cut(df[x_col], bins=edges, labels=labels, include_lowest=True)
    df["_is_nan"] = df[y_col].isna().astype(float)

    agg = (
        df.groupby("_bin", observed=True)
        .agg(nan_rate=("_is_nan", "mean"), count=("_is_nan", "count"))
        .reset_index()
        .rename(columns={"_bin": "bin_mid"})
    )
    agg["bin_mid"] = agg["bin_mid"].astype(float)
    agg = agg[agg["count"] >= min_count]
    return agg


def slice_table(
    df: pd.DataFrame,
    ir_center: float = 0.5,
    gr_center: float = 0.5,
    tol: float = 0.05,
    max_rows: int = 20,
) -> pd.DataFrame:
    """Return a human-readable table of sample rows near (IR≈ir_center, GR≈gr_center).

    Includes: confusion-matrix counts, ir, gr, tpr_i, tpr_j, eod.

    Parameters
    ----------
    df : pd.DataFrame
        Full annotated DataFrame (must contain ``eod`` column).
    ir_center, gr_center : float
        Target centre values for IR and GR (default 0.5 each).
    tol : float
        Half-width of the window (default 0.05, i.e. ±5 pp).
    max_rows : int
        Maximum number of rows to return (default 20).
    """
    sub = _filter_near(df, "ir", ir_center, tol)
    sub = _filter_near(sub, "gr", gr_center, tol)

    display_cols = [
        "i_tp", "i_fp", "i_tn", "i_fn",
        "j_tp", "j_fp", "j_tn", "j_fn",
        "ir", "gr", "tpr_i", "tpr_j", "eod",
    ]
    # tpr_i / tpr_j may or may not exist yet; add them if missing
    if "tpr_i" not in sub.columns:
        sub = sub.copy()
        denom_i = sub["i_tp"] + sub["i_fn"]
        denom_j = sub["j_tp"] + sub["j_fn"]
        sub["tpr_i"] = (sub["i_tp"] / denom_i.where(denom_i != 0, other=np.nan)).round(4)
        sub["tpr_j"] = (sub["j_tp"] / denom_j.where(denom_j != 0, other=np.nan)).round(4)

    existing = [c for c in display_cols if c in sub.columns]
    return sub[existing].head(max_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure factories
# ---------------------------------------------------------------------------

def _make_eod_figure(
    agg: pd.DataFrame,
    x_label: str,
    title: str,
) -> go.Figure:
    """Shared builder for EOD mean±std band plots."""
    x = agg["bin_mid"].values
    y = agg["mean"].values
    std = agg["std"].fillna(0).values

    fig = go.Figure()
    # Shaded ±1 std band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y + std, (y - std)[::-1]]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="±1 std",
        )
    )
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color="rgb(99,110,250)", width=2),
            name="mean EOD",
        )
    )
    # Reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="EOD (TPR_j − TPR_i)",
        yaxis=dict(range=[-1.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


def plot_eod_vs_ir(
    df: pd.DataFrame,
    gr_center: float = 0.5,
    gr_tol: float = 0.05,
    n_bins: int = _DEFAULT_BINS,
) -> go.Figure:
    """EOD mean ± std vs IR, for rows near GR ≈ gr_center.

    Parameters
    ----------
    df : pd.DataFrame
        Annotated DataFrame with ``eod``, ``ir``, ``gr`` columns.
    gr_center : float
        Centre of the GR slice (default 0.5).
    gr_tol : float
        Half-width of the GR window (default ±0.05).
    n_bins : int
        Number of IR bins (default 20).
    """
    sub = _filter_near(df, "gr", gr_center, gr_tol).dropna(subset=["eod"])
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data in GR slice", showarrow=False)
        return fig

    agg = _bin_and_summarise(sub, "ir", "eod", n_bins=n_bins)
    return _make_eod_figure(
        agg,
        x_label="Imbalance Ratio (IR)",
        title=f"EOD vs IR  |  GR ≈ {gr_center:.2f} ± {gr_tol:.2f}",
    )


def plot_eod_vs_gr(
    df: pd.DataFrame,
    ir_center: float = 0.5,
    ir_tol: float = 0.05,
    n_bins: int = _DEFAULT_BINS,
) -> go.Figure:
    """EOD mean ± std vs GR, for rows near IR ≈ ir_center.

    Parameters
    ----------
    df : pd.DataFrame
        Annotated DataFrame with ``eod``, ``ir``, ``gr`` columns.
    ir_center : float
        Centre of the IR slice (default 0.5).
    ir_tol : float
        Half-width of the IR window (default ±0.05).
    n_bins : int
        Number of GR bins (default 20).
    """
    sub = _filter_near(df, "ir", ir_center, ir_tol).dropna(subset=["eod"])
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data in IR slice", showarrow=False)
        return fig

    agg = _bin_and_summarise(sub, "gr", "eod", n_bins=n_bins)
    return _make_eod_figure(
        agg,
        x_label="Group Ratio (GR)",
        title=f"EOD vs GR  |  IR ≈ {ir_center:.2f} ± {ir_tol:.2f}",
    )


def _make_nan_figure(
    agg: pd.DataFrame,
    x_label: str,
    title: str,
) -> go.Figure:
    """Shared builder for NaN-rate line plots."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["bin_mid"].values,
            y=agg["nan_rate"].values,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color="rgb(239,85,59)", width=2),
            name="NaN rate",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="NaN rate",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(t=60, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


def plot_nan_vs_ir(
    df: pd.DataFrame,
    gr_center: float = 0.5,
    gr_tol: float = 0.05,
    n_bins: int = _DEFAULT_BINS,
) -> go.Figure:
    """NaN rate of EOD vs IR, for rows near GR ≈ gr_center."""
    sub = _filter_near(df, "gr", gr_center, gr_tol)
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data in GR slice", showarrow=False)
        return fig

    agg = _nan_rate_by_bin(sub, "ir", "eod", n_bins=n_bins)
    return _make_nan_figure(
        agg,
        x_label="Imbalance Ratio (IR)",
        title=f"EOD NaN rate vs IR  |  GR ≈ {gr_center:.2f} ± {gr_tol:.2f}",
    )


def plot_nan_vs_gr(
    df: pd.DataFrame,
    ir_center: float = 0.5,
    ir_tol: float = 0.05,
    n_bins: int = _DEFAULT_BINS,
) -> go.Figure:
    """NaN rate of EOD vs GR, for rows near IR ≈ ir_center."""
    sub = _filter_near(df, "ir", ir_center, ir_tol)
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data in IR slice", showarrow=False)
        return fig

    agg = _nan_rate_by_bin(sub, "gr", "eod", n_bins=n_bins)
    return _make_nan_figure(
        agg,
        x_label="Group Ratio (GR)",
        title=f"EOD NaN rate vs GR  |  IR ≈ {ir_center:.2f} ± {ir_tol:.2f}",
    )
