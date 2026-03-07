"""Optional place for user-defined metrics.

This file is imported by ``app.py`` after the built-in metrics are registered.
To add a new metric, uncomment the example below, edit it, and reload Streamlit.
"""

from __future__ import annotations

# from metric_registry import MetricSpec, register_metric, safe_divide
# import pandas as pd
# import numpy as np
#
#
# def balanced_outcome_gap(df: pd.DataFrame) -> np.ndarray:
#     """Example custom metric: average of TPR gap and TNR gap, using j - i sign."""
#
#     tpr_i = safe_divide(df["i_tp"], df["i_tp"] + df["i_fn"])
#     tpr_j = safe_divide(df["j_tp"], df["j_tp"] + df["j_fn"])
#     tnr_i = safe_divide(df["i_tn"], df["i_tn"] + df["i_fp"])
#     tnr_j = safe_divide(df["j_tn"], df["j_tn"] + df["j_fp"])
#     return 0.5 * ((tpr_j - tpr_i) + (tnr_j - tnr_i))
#
#
# register_metric(
#     MetricSpec(
#         key="balanced_outcome_gap",
#         label="Balanced Outcome Gap",
#         category="fairness",
#         description="Example extension metric; meant as a template for experimentation.",
#         formula="0.5 * ((j_tpr - i_tpr) + (j_tnr - i_tnr))",
#         compute=balanced_outcome_gap,
#     )
# )
