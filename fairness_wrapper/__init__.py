"""
fairness_wrapper
================
A thin, modular pipeline for studying how fairness metrics behave
across varying class-imbalance ratios (IR) and protected-group ratios (GR).

Public API
----------
generate_cm_pairs   – generate all feasible confusion-matrix pairs for budget n
compute_eod         – compute Equal Opportunity Difference with a NaN policy
wrap                – wrapper interface (identity for now)
plot_eod_vs_ir      – EOD vs IR at a fixed GR slice
plot_eod_vs_gr      – EOD vs GR at a fixed IR slice
plot_nan_vs_ir      – NaN rate vs IR
plot_nan_vs_gr      – NaN rate vs GR
slice_table         – example rows near (IR≈0.5, GR≈0.5)
"""

from .generate import generate_cm_pairs
from .metrics import compute_eod, nan_rate
from .wrapper import wrap
from .plots import (
    plot_eod_vs_ir,
    plot_eod_vs_gr,
    plot_nan_vs_ir,
    plot_nan_vs_gr,
    slice_table,
)

__all__ = [
    "generate_cm_pairs",
    "compute_eod",
    "nan_rate",
    "wrap",
    "plot_eod_vs_ir",
    "plot_eod_vs_gr",
    "plot_nan_vs_ir",
    "plot_nan_vs_gr",
    "slice_table",
]
