"""
wrapper.py
----------
Defines the wrapper interface for fairness metrics.

The wrapper is a post-processing transform W(·) applied to a raw metric
value.  Its purpose is to make metric *interpretation* more stable across
varying imbalance ratio (IR) and group ratio (GR).

Current implementation
----------------------
The wrapper is the **identity function**: it returns the metric value
unchanged.  The interface and argument signature are final — only the
implementation body will change in future iterations.

How to extend
-------------
Replace the body of ``wrap()`` with a correction formula, e.g.:

    # Example: feasible-range normalisation (future work)
    lo, hi = feasible_range(cm_i, cm_j, ir=ir, gr=gr)
    if hi == lo:
        return 0.0          # degenerate case
    return 2 * (metric_value - lo) / (hi - lo) - 1  # rescale to [-1, 1]

The function signature must remain stable so that the Streamlit UI and
any downstream code continue to work without modification.
"""

from __future__ import annotations

def wrap(
    metric_value: float,
    cm_i: dict[str, int],
    cm_j: dict[str, int],
    *,
    ir: float,
    gr: float,
    policy: str,
) -> float:
    """Apply the fairness-metric wrapper to *metric_value*.

    Parameters
    ----------
    metric_value : float
        The raw fairness metric (e.g., EOD) computed for a single row.
        May be NaN.
    cm_i : dict
        Confusion-matrix counts for the protected group *i*.
        Keys: "tp", "fp", "tn", "fn".
    cm_j : dict
        Confusion-matrix counts for the unprotected group *j*.
        Keys: "tp", "fp", "tn", "fn".
    ir : float
        Imbalance ratio for this row (positive-class fraction, ∈ [0, 1]).
    gr : float
        Group ratio for this row (majority-group fraction, ∈ [0, 1]).
    policy : str
        The NaN-handling policy that was used to compute *metric_value*.
        Provided for context; the wrapper may choose to behave differently
        depending on the policy.

    Returns
    -------
    float
        Wrapped metric value.  Currently identical to *metric_value*.

    Notes
    -----
    This is the **extension point** for your MSc thesis.  Planned next steps:
    1. Compute the *feasible range* [lo, hi] of EOD given (cm_i, cm_j, ir, gr).
    2. Normalise: W = 2 * (EOD − lo) / (hi − lo) − 1 → ∈ [−1, 1].
    3. Add an SR (selection-rate) dimension if needed.
    """
    # ------------------------------------------------------------------ #
    #  IDENTITY – replace this block to implement a real wrapper           #
    # ------------------------------------------------------------------ #
    return metric_value

