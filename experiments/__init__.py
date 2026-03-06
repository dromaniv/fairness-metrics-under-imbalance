"""
experiments
===========
Interactive exploration of group fairness metrics under class-imbalance
and protected-group-size variation.

Public API
----------
generate_cm_pairs       – enumerate all feasible CM pairs for a budget n
load_cm_pairs           – load a pre-generated .bin file
compute_metric          – compute any supported metric (NaN-policy aware)
compute_eod             – backward-compat alias
nan_rate                – fraction of NaN values
perfect_fairness_rate   – fraction where |value| ≤ eps
METRIC_LABELS           – dict  key → short human-readable label
METRIC_RANGE            – dict  key → (lo, hi) theoretical range
METRIC_SLICE_DEFAULTS   – dict  key → (ir_center, gr_center)
apply_jeffreys_df       – Jeffreys smoothing only (stage 1, no arcsin)
apply_wrapper_df        – Jeffreys smoothing + arcsin VST (stage 1+2)
wrap                    – Jeffreys-VST wrapper for a single CM pair
WRAPPER_METRICS         – tuple of metric keys accepted by the wrapper
plot_metric_vs_ir       – metric mean±std vs IR at a fixed GR slice
plot_metric_vs_gr       – metric mean±std vs GR at a fixed IR slice
plot_nan_vs_ir          – NaN rate vs IR
plot_nan_vs_gr          – NaN rate vs GR
slice_table             – example rows near a chosen (IR, GR) centre
"""

from .generate import generate_cm_pairs, load_cm_pairs
from .metrics  import (
    compute_metric, compute_eod, nan_rate, perfect_fairness_rate,
    METRIC_LABELS, METRIC_RANGE, METRIC_SLICE_DEFAULTS,
)
from .wrapper  import apply_jeffreys_df, apply_wrapper_df, wrap, WRAPPER_METRICS
from .plots    import (
    plot_eod_vs_ir,
    plot_eod_vs_gr,
    plot_nan_vs_ir,
    plot_nan_vs_gr,
    slice_table,
)

__all__ = [
    "generate_cm_pairs",
    "load_cm_pairs",
    "compute_metric",
    "compute_eod",
    "nan_rate",
    "perfect_fairness_rate",
    "METRIC_LABELS",
    "METRIC_RANGE",
    "METRIC_SLICE_DEFAULTS",
    "apply_jeffreys_df",
    "apply_wrapper_df",
    "wrap",
    "WRAPPER_METRICS",
    "plot_eod_vs_ir",
    "plot_eod_vs_gr",
    "plot_nan_vs_ir",
    "plot_nan_vs_gr",
    "slice_table",
]
