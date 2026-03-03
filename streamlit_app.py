"""
streamlit_app.py
----------------
Interactive dashboard for exploring how Equal Opportunity Difference (EOD)
varies with class imbalance ratio (IR) and protected-group ratio (GR).

Run with:
    streamlit run streamlit_app.py

Layout
------
Sidebar  – controls: sample size n, NaN policy, alpha, wrapper toggle, slice centres
Main     – 4 Plotly charts in a 2×2 grid + example data table
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fairness Wrapper Explorer",
    page_icon="⚖️",
    layout="wide",
)

from fairness_wrapper import (  # noqa: E402 – after set_page_config
    generate_cm_pairs,
    compute_eod,
    wrap,
    plot_eod_vs_ir,
    plot_eod_vs_gr,
    plot_nan_vs_ir,
    plot_nan_vs_gr,
    slice_table,
    nan_rate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating confusion-matrix pairs …")
def cached_generate(n: int) -> pd.DataFrame:
    """Generate and cache the confusion-matrix pair dataset for budget *n*.

    Results are cached per value of *n* so switching between sizes is instant
    after the first load.
    """
    return generate_cm_pairs(n=n)


def apply_wrapper(df: pd.DataFrame, policy: str) -> pd.Series:
    """Apply wrap() to the EOD column and return the wrapped series.

    While wrap() is the identity function this is a vectorised no-op —
    it simply returns the existing EOD series unchanged, which is O(1).

    When you replace the body of wrap() with a real correction formula
    that needs per-row context (cm_i, cm_j, ir, gr), swap this to the
    row-wise implementation below:

        results = []
        for _, row in df.iterrows():
            cm_i = {"tp": row.i_tp, "fp": row.i_fp, "tn": row.i_tn, "fn": row.i_fn}
            cm_j = {"tp": row.j_tp, "fp": row.j_fp, "tn": row.j_tn, "fn": row.j_fn}
            results.append(
                wrap(row["eod"], cm_i, cm_j, ir=row["ir"], gr=row["gr"], policy=policy)
            )
        return pd.Series(results, index=df.index, name="eod_wrapped")
    """
    # Identity fast-path: wrap() returns its input unchanged, so no looping needed.
    return df["eod"].rename("eod_wrapped")


# ---------------------------------------------------------------------------
# Sidebar – controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ Fairness Wrapper Explorer")
    st.markdown("---")

    st.subheader("Data")
    n = st.slider(
        "Sample size n  (budget per row)",
        min_value=8,
        max_value=40,
        value=24,
        step=4,
        help=(
            "Total count summed over all 8 cells.  "
            "n=24 → ~2 500 rows (fast).  n=40 → ~~52 000 rows (a few seconds)."
        ),
    )

    st.subheader("Metric")
    metric = st.selectbox(
        "Fairness metric",
        options=["Equal Opportunity Difference (EOD)"],
        index=0,
        help="More metrics will be added in future iterations.",
    )

    st.subheader("NaN policy")
    policy = st.radio(
        "How to handle undefined TPR (denominator = 0)",
        options=["nan", "zero", "smooth_alpha"],
        index=0,
        format_func=lambda p: {
            "nan": "nan  – propagate NaN (mathematically honest)",
            "zero": "zero – impute 0 (convention; biases EOD)",
            "smooth_alpha": "smooth_alpha – Laplace smoothing (no NaNs)",
        }[p],
    )

    alpha = 1.0
    if policy == "smooth_alpha":
        alpha = st.number_input(
            "Alpha (smoothing strength)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="α > 0.  α=1 is the Laplace (uniform) prior.",
        )

    st.subheader("Wrapper")
    wrapper_on = st.toggle(
        "Apply wrapper W(·)",
        value=False,
        help=(
            "Currently the wrapper is the identity function and has no effect. "
            "Toggle exists for future use when a real correction is implemented."
        ),
    )

    st.markdown("---")
    st.subheader("Slice centres")
    gr_center = st.slider("GR centre (for EOD vs IR plot)", 0.0, 1.0, 0.5, 0.05)
    ir_center = st.slider("IR centre (for EOD vs GR plot)", 0.0, 1.0, 0.5, 0.05)
    slice_tol = st.slider("Slice tolerance (±)", 0.02, 0.20, 0.05, 0.01)

    st.markdown("---")
    st.caption(
        "Thesis MVP – fairness metrics under imbalance.  "
        "Wrapper: identity (next step: feasible-range normalisation)."
    )

# ---------------------------------------------------------------------------
# Data generation & metric computation
# ---------------------------------------------------------------------------

df = cached_generate(n)

eod_series = compute_eod(df, policy=policy, alpha=alpha)
df = df.copy()
df["eod"] = eod_series

# Compute tpr_i / tpr_j for the table (using the same policy)
from fairness_wrapper.metrics import _tpr  # noqa: E402 – internal import

df["tpr_i"] = _tpr(df["i_tp"], df["i_fn"], policy=policy, alpha=alpha).round(4)
df["tpr_j"] = _tpr(df["j_tp"], df["j_fn"], policy=policy, alpha=alpha).round(4)

if wrapper_on:
    df["eod"] = apply_wrapper(df, policy=policy)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Fairness Metrics Under Imbalance – Wrapper Explorer")

# ── Invariance helpers ───────────────────────────────────────────────────────
def _badge(invariant: bool, true_label: str = "True", false_label: str = "False") -> str:
    """Return an HTML badge: green for invariant=True, red for False."""
    if invariant:
        return (
            f'<span style="background:#1e7e34;color:white;padding:2px 10px;'
            f'border-radius:12px;font-weight:600;font-size:0.85em;">✔ {true_label}</span>'
        )
    return (
        f'<span style="background:#c0392b;color:white;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">✘ {false_label}</span>'
    )

# Invariance criteria (thresholds are intentionally strict so they are False by default):
#   NaN-rate invariant  → NaN rate < 1 %  (metric defined for ≥99 % of rows)
#   Spread invariant    → std of |EOD| across IR/GR bins < 0.05  (metric barely moves)
#   Wrapper invariant   → wrapper is ON *and* not identity
#                         (currently always False – placeholder for real correction)
_eod_defined = df["eod"].dropna()
_nan_inv  = nan_rate(df["eod"]) < 0.01
_spread_inv = float(_eod_defined.std()) < 0.05 if len(_eod_defined) > 1 else True
_wrap_inv = wrapper_on and False   # identity wrapper never achieves invariance

# ── Summary bar ─────────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Total rows", f"{len(df):,}")
    st.markdown(_badge(True, "Fixed", "Fixed"), unsafe_allow_html=True)
    st.caption("Always equal to C(n+7,7)")

with col_b:
    st.metric("EOD NaN rate", f"{nan_rate(df['eod']):.1%}")
    st.markdown(_badge(_nan_inv, "Invariant", "Variant"), unsafe_allow_html=True)
    st.caption("Invariant if NaN rate < 1 %")

with col_c:
    st.metric("Mean |EOD|", f"{_eod_defined.abs().mean():.4f}" if len(_eod_defined) else "n/a")
    st.markdown(_badge(_spread_inv, "Invariant", "Variant"), unsafe_allow_html=True)
    st.caption("Invariant if std(|EOD|) < 0.05")

with col_d:
    st.metric("Wrapper", "ON (identity)" if wrapper_on else "OFF")
    st.markdown(_badge(_wrap_inv, "Invariant", "Not correcting"), unsafe_allow_html=True)
    st.caption("Invariant only when a real correction is applied")

st.markdown("---")

# ── 2×2 plots ───────────────────────────────────────────────────────────────
st.subheader("Diagnostic Plots")

row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

with row1_left:
    fig1 = plot_eod_vs_ir(df, gr_center=gr_center, gr_tol=slice_tol)
    st.plotly_chart(fig1, width="stretch")

with row1_right:
    fig2 = plot_eod_vs_gr(df, ir_center=ir_center, ir_tol=slice_tol)
    st.plotly_chart(fig2, width="stretch")

with row2_left:
    fig3 = plot_nan_vs_ir(df, gr_center=gr_center, gr_tol=slice_tol)
    st.plotly_chart(fig3, width="stretch")

with row2_right:
    fig4 = plot_nan_vs_gr(df, ir_center=ir_center, ir_tol=slice_tol)
    st.plotly_chart(fig4, width="stretch")

# ── Example table ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(
    f"Example rows  |  IR ≈ {ir_center:.2f} ± {slice_tol:.2f},  "
    f"GR ≈ {gr_center:.2f} ± {slice_tol:.2f}"
)

table_df = slice_table(df, ir_center=ir_center, gr_center=gr_center, tol=slice_tol)
if table_df.empty:
    st.warning(
        "No rows match the current slice.  "
        "Try increasing the tolerance or adjusting the slice centres."
    )
else:
    # Round floats for readability
    float_cols = [c for c in ["ir", "gr", "tpr_i", "tpr_j", "eod"] if c in table_df.columns]
    st.dataframe(
        table_df.style.format({c: "{:.4f}" for c in float_cols}),
        width="stretch",
    )
    st.caption(
        f"Showing up to 20 rows out of "
        f"{len(df[(np.abs(df['ir'] - ir_center) <= slice_tol) & (np.abs(df['gr'] - gr_center) <= slice_tol)]):,} "
        f"matching rows."
    )

# ── Footer note ──────────────────────────────────────────────────────────────
with st.expander("ℹ️  Definitions & conventions"):
    st.markdown(
        """
**Equal Opportunity Difference (EOD)**
$$\\text{EOD} = \\text{TPR}_j - \\text{TPR}_i = \\frac{\\text{TP}_j}{\\text{TP}_j+\\text{FN}_j} - \\frac{\\text{TP}_i}{\\text{TP}_i+\\text{FN}_i}$$

- **Positive value** → unprotected group (*j*) has higher recall (favours majority).
- **Zero** → equal opportunity (perfect fairness for this metric).
- **Negative value** → protected group (*i*) has higher recall.
- Theoretical range when defined: **[−1, +1]**.

**Imbalance Ratio (IR)**
$$\\text{IR} = \\frac{\\text{TP}_i + \\text{FN}_i + \\text{TP}_j + \\text{FN}_j}{n}$$
Share of actual positives in the whole sample.  IR = 0.5 → balanced classes.

**Group Ratio (GR)**
$$\\text{GR} = \\frac{\\text{TP}_j + \\text{FP}_j + \\text{TN}_j + \\text{FN}_j}{n}$$
Share of the unprotected group in the sample.  GR = 0.5 → equal group sizes.

**NaN policies**

| Policy | Behaviour when TP+FN = 0 |
|--------|--------------------------|
| `nan` | propagate NaN (mathematically honest) |
| `zero` | impute TPR = 0 (convention; can bias EOD) |
| `smooth_alpha` | TPR = (TP+α)/(TP+FN+2α) — Laplace smoothing, no NaNs |

**Wrapper W(·)** – currently the identity function.
Next step: feasible-range normalisation.
        """
    )



