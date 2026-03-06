"""
experiments/app.py
------------------
Streamlit dashboard for exploring fairness metrics under varying
class imbalance (IR) and protected-group size (GR).

    streamlit run experiments/app.py
"""

from __future__ import annotations

import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fairness Metrics Under Imbalance",
    page_icon="⚖️",
    layout="wide",
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments import (  # noqa: E402
    generate_cm_pairs,
    load_cm_pairs,
    compute_metric,
    apply_jeffreys_df,
    apply_wrapper_df,
    plot_eod_vs_ir,
    plot_eod_vs_gr,
    plot_nan_vs_ir,
    plot_nan_vs_gr,
    slice_table,
    nan_rate,
    perfect_fairness_rate,
    METRIC_LABELS,
    METRIC_RANGE,
    METRIC_SLICE_DEFAULTS,
    WRAPPER_METRICS,
)
from experiments.metrics import _tpr  # noqa: E402

_CAF_METRICS = {"caf_q_rms_ha", "caf_q_rms_ha_abs", "caf_eopp_q_ha"}
_NO_WRAPPER  = _CAF_METRICS

_RATIO_FRACS  = [1/12, 1/4, 1/2, 3/4, 11/12]
_RATIO_LABELS = ["1/12", "1/4", "1/2", "3/4", "11/12"]


def _auto_hist_params(n: int) -> tuple[int, float]:
    n_bins = max(8, min(40, n * 2))
    tol = max(0.02, round(0.5 / n, 3))
    return n_bins, tol


@st.cache_data(show_spinner="Generating confusion-matrix pairs …", max_entries=8)
def _cached_generate(n: int) -> pd.DataFrame:
    return generate_cm_pairs(n=n)


@st.cache_data(show_spinner="Loading confusion-matrix pairs from file …", max_entries=4)
def _cached_load_path(path: str) -> pd.DataFrame:
    return load_cm_pairs(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Controls")

    st.markdown("### Dataset")
    bin_path = st.text_input(
        "Path to .bin file  (leave empty to generate)",
        value="",
        placeholder="/path/to/Set(08,56).bin",
        help="Path to a pre-generated .bin file. Loaded directly from disk, no size limit. "
             "n is detected from the row sum.",
    )

    _source = "path" if bin_path.strip() else "generate"

    if _source == "generate":
        n = st.slider("Sample size n", min_value=8, max_value=32, value=24, step=4,
                      help="n=24 → ~2 500 rows, n=32 → ~18 000 rows.")
    else:
        n = None

    st.markdown("---")

    st.markdown("### Metric")
    metric = st.selectbox(
        "Fairness metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
    )

    _is_caf = metric in _CAF_METRICS
    if _is_caf:
        kappa = st.slider(
            "k smoothing (Haldane–Anscombe)",
            min_value=0.0, max_value=2.0, value=0.5, step=0.05,
            help="Continuity correction added to each cell in the CAF 2×2 table. "
                 "k=0.5 is the Haldane–Anscombe default.",
        )
    else:
        kappa = 0.5

    st.markdown("---")

    st.markdown("### Wrapper")
    _wrapper_applicable = metric not in _NO_WRAPPER

    jeffreys_on = st.toggle(
        "Jeffreys smoothing  (+½)",
        value=False,
        disabled=not _wrapper_applicable,
        help="Replaces raw rates with Jeffreys-posterior means: "
             "r̃ = (x + ½) / (d + 1). Always defined, eliminates 0/0, "
             "shrinks extreme estimates towards ½. "
             "Useful when counts are small or denominators can be zero.",
    )

    vst_on = st.toggle(
        "VST  (arcsin transform)",
        value=False,
        disabled=(not _wrapper_applicable) or (not jeffreys_on),
        help="Applies the arcsine variance-stabilising transform on top of Jeffreys rates: "
             "W = (2/π)·(arcsin(√r̃ⱼ) − arcsin(√r̃ᵢ)) ∈ [−1, 1]. "
             "Stabilises variance across the full [0,1] range, making differences "
             "at extreme rates (near 0 or 1) comparable to those near 0.5. "
             "Requires Jeffreys smoothing.",
    )

    st.markdown("---")

    st.markdown("### Slice centres")
    _def_ir, _def_gr = METRIC_SLICE_DEFAULTS.get(metric, (0.5, 0.5))
    gr_center = st.slider("GR centre", 0.0, 1.0, _def_gr, 0.05)
    ir_center = st.slider("IR centre", 0.0, 1.0, _def_ir, 0.05)
    slice_tol = st.slider("Tolerance (±)", 0.02, 0.20, 0.05, 0.01)


# ---------------------------------------------------------------------------
# Load / generate
# ---------------------------------------------------------------------------

if _source == "path":
    _resolved = bin_path.strip()
    if not os.path.isabs(_resolved):
        _resolved = os.path.join(_ROOT, _resolved)
    if not os.path.exists(_resolved):
        st.error(f"File not found: `{_resolved}`")
        st.stop()
    _base = _cached_load_path(_resolved)
    n = int(_base[["i_tp","i_fp","i_tn","i_fn",
                    "j_tp","j_fp","j_tn","j_fn"]].sum(axis=1).iloc[0])
    st.sidebar.success(f"Loaded — detected **n = {n}**")
else:
    _base = _cached_generate(n)

# compute the metric value without mutating the cached DataFrame
_value = compute_metric(_base, metric=metric, policy="nan", kappa=kappa)

if _wrapper_applicable:
    if vst_on and jeffreys_on:
        _value = apply_wrapper_df(_base, metric=metric)
    elif jeffreys_on:
        _value = apply_jeffreys_df(_base, metric=metric)

_tpr_i = _tpr(_base["i_tp"], _base["i_fn"], policy="nan", alpha=1.0).round(4)
_tpr_j = _tpr(_base["j_tp"], _base["j_fn"], policy="nan", alpha=1.0).round(4)

# build a thin working frame only for slicing / plotting
df = _base.assign(value=_value, tpr_i=_tpr_i, tpr_j=_tpr_j)

_short = METRIC_LABELS[metric]
if _wrapper_applicable and vst_on and jeffreys_on:
    _suffix = " (Jeffreys + VST)"
elif _wrapper_applicable and jeffreys_on:
    _suffix = " (Jeffreys)"
else:
    _suffix = ""
_y_label = _short + _suffix

_hist_bins, _hist_tol = _auto_hist_params(n)
_lo, _hi = METRIC_RANGE.get(metric, (-1.0, 1.0))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_defined = df["value"].dropna()
_pf = perfect_fairness_rate(df["value"], eps=0.0)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total rows", f"{len(df):,}")
col_b.metric("NaN rate", f"{nan_rate(df['value']):.1%}")
col_c.metric(f"Mean |{_short}|", f"{_defined.abs().mean():.4f}" if len(_defined) else "n/a")
col_d.metric("Perfect fairness", f"{_pf:.1%}",
             help=f"Fraction of rows where |value| = 0 (n={n}).")

st.markdown("---")

# ---------------------------------------------------------------------------
# Distribution grid
# ---------------------------------------------------------------------------

st.subheader("Distribution across IR × GR grid")

_header = st.columns([1] + [4] * 5)
_header[0].markdown("IR \\ GR")
for _gi, _gl in enumerate(_RATIO_LABELS):
    _header[_gi + 1].markdown(f"**GR = {_gl}**")

for _ir_v, _ir_l in zip(reversed(_RATIO_FRACS), reversed(_RATIO_LABELS)):
    row_cols = st.columns([1] + [4] * 5)
    row_cols[0].markdown(f"**IR = {_ir_l}**")

    for _gi, _gr_v in enumerate(_RATIO_FRACS):
        _mask = (np.abs(df["ir"] - _ir_v) <= _hist_tol) & (np.abs(df["gr"] - _gr_v) <= _hist_tol)
        _vals = df.loc[_mask, "value"].dropna()
        _total = int(_mask.sum())
        _nan_frac = 1.0 - len(_vals) / _total if _total > 0 else 1.0

        fig, ax = plt.subplots(figsize=(3, 1.8))
        if len(_vals):
            _lo_v = _lo if _lo is not None else float(_vals.min())
            _hi_v = _hi if _hi is not None else float(_vals.max())
            ax.hist(_vals, bins=np.linspace(_lo_v, _hi_v, _hist_bins + 1),
                    color="black", edgecolor="black", linewidth=0.3)
        if _nan_frac > 0:
            ax.text(0.97, 0.93, f"undef {_nan_frac:.0%}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="red")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout(pad=0.2)
        row_cols[_gi + 1].pyplot(fig, width="content")
        plt.close(fig)
        gc.collect()

st.markdown("---")

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

st.subheader("Diagnostic plots")

_n_bins = _hist_bins

r1l, r1r = st.columns(2)
r2l, r2r = st.columns(2)

with r1l:
    st.plotly_chart(
        plot_eod_vs_ir(df, gr_center=gr_center, gr_tol=slice_tol,
                       n_bins=_n_bins, y_col="value", y_label=_y_label,
                       y_range=(_lo, _hi)),
        width="stretch",
    )
with r1r:
    st.plotly_chart(
        plot_eod_vs_gr(df, ir_center=ir_center, ir_tol=slice_tol,
                       n_bins=_n_bins, y_col="value", y_label=_y_label,
                       y_range=(_lo, _hi)),
        width="stretch",
    )
with r2l:
    st.plotly_chart(
        plot_nan_vs_ir(df, gr_center=gr_center, gr_tol=slice_tol,
                       n_bins=_n_bins, y_col="value"),
        width="stretch",
    )
with r2r:
    st.plotly_chart(
        plot_nan_vs_gr(df, ir_center=ir_center, ir_tol=slice_tol,
                       n_bins=_n_bins, y_col="value"),
        width="stretch",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------

st.subheader(
    f"Sample rows  ·  IR ≈ {ir_center:.2f} ± {slice_tol:.2f}"
    f"  ·  GR ≈ {gr_center:.2f} ± {slice_tol:.2f}"
)

table = slice_table(df, ir_center=ir_center, gr_center=gr_center, tol=slice_tol)
if table.empty:
    st.warning("No rows in this slice — try increasing the tolerance.")
else:
    float_cols = [c for c in ["ir", "gr", "tpr_i", "tpr_j", "value"] if c in table.columns]
    st.dataframe(
        table.style.format({c: "{:.4f}" for c in float_cols}),
        width="stretch",
    )
    n_match = int(
        ((np.abs(df["ir"] - ir_center) <= slice_tol) &
         (np.abs(df["gr"] - gr_center) <= slice_tol)).sum()
    )
    st.caption(f"Showing up to 20 of {n_match:,} matching rows.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

with st.expander("Metric & wrapper definitions"):
    st.markdown(r"""
### Metrics

| Key | Name | Formula | Range |
|-----|------|---------|-------|
| `accd` | Accuracy Equality          | ACC_j − ACC_i                   | [−1, 1] |
| `spd`  | Statistical Parity         | SR_j − SR_i                     | [−1, 1] |
| `eod`  | Equal Opportunity          | TPR_j − TPR_i                   | [−1, 1] |
| `fprd` | Predictive Equality        | FPR_j − FPR_i                   | [−1, 1] |
| `eqod` | Equalized Odds             | max(\|TPR diff\|, \|FPR diff\|) | [0, 1]  |
| `ppvd` | Positive Predictive Parity | PPV_j − PPV_i                   | [−1, 1] |
| `npvd` | Negative Predictive Parity | NPV_j − NPV_i                   | [−1, 1] |
| `caf_q_rms_ha`     | CAF-Q (unsigned) | sqrt((Q₀²+Q₁²)/2)              | [0, 1)  |
| `caf_q_rms_ha_abs` | CAF-Q (signed)   | sign(Q₀+Q₁)·sqrt((Q₀²+Q₁²)/2) | (−1, 1) |
| `caf_eopp_q_ha`    | CAF Equal Opportunity | \|Q₁\|                   | [0, 1)  |

`i` = protected group, `j` = unprotected group. Positive value → group j has the higher rate. Zero = perfect fairness.

`eqod` is zero only when both TPR and FPR differences are simultaneously zero.

---

### CAF-Q

For each stratum $y \in \{0, 1\}$, form the 2×2 table of $(\hat{Y}, S)$ within $Y=y$:

$$\text{cell layout: } a_y = n_{y,\hat{0},i},\; b_y = n_{y,\hat{0},j},\; c_y = n_{y,\hat{1},i},\; d_y = n_{y,\hat{1},j}$$

Apply Haldane–Anscombe correction $k$, compute the smoothed odds ratio and Yule's Q:

$$OR_y = \frac{(a_y+k)(d_y+k)}{(b_y+k)(c_y+k)}, \qquad Q_y = \frac{OR_y - 1}{OR_y + 1}$$

**Unsigned** (`caf_q_rms_ha`): $\;\sqrt{(Q_0^2+Q_1^2)/2} \in [0,1)$

**Signed** (`caf_q_rms_ha_abs`): $\;\operatorname{sign}(Q_0+Q_1)\cdot\sqrt{(Q_0^2+Q_1^2)/2} \in (-1,1)$

**Equal Opportunity only** (`caf_eopp_q_ha`): $\;|Q_1|$ — Y=1 stratum only.

Always finite for $k > 0$. Zero iff association between $\hat{Y}$ and $S$ is zero in both strata.

---

### Wrapper

**Jeffreys smoothing** replaces raw rates with posterior means under a Beta(½, ½) prior:

$$\tilde{r}_a = \frac{x_a + \tfrac{1}{2}}{d_a + 1}$$

Always in $(0, 1)$ — no 0/0, estimates at the boundary are shrunk toward ½.

**VST** (requires Jeffreys) applies the arcsine variance-stabilising transform:

$$W = \frac{2}{\pi}\!\left(\arcsin\sqrt{\tilde{r}_j} - \arcsin\sqrt{\tilde{r}_i}\right) \in [-1, 1]$$

$\operatorname{Var}(\arcsin\sqrt{\hat{p}}) \approx 1/(4n)$ regardless of $p$.

Applies to all classic and Equalized Odds metrics. Not applicable to CAF (uses its own $k$ correction).

---

### Dataset

`.bin` files are pickled `(m, 8)` int8 arrays from `sets_creation.py`:

```
python sets_creation.py -g 8 56 -b Set\(08,56\).bin
```

$n$ is auto-detected from the row sum. Enter the path in the sidebar — loaded directly from disk, no size limit.
    """)

