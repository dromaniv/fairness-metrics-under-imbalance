"""
generate.py
-----------
Generates all feasible pairs of group-level confusion matrices whose
8 cell counts (i_tp, i_fp, i_tn, i_fn, j_tp, j_fp, j_tn, j_fn) sum
exactly to a given budget *n*.

The enumeration relies on the combinatorial simplex generator from
``sets_creation.py`` in the parent project.

Usage
-----
>>> from fairness_wrapper.generate import generate_cm_pairs
>>> df = generate_cm_pairs(n=24)   # ~2 500 rows for n=24, fast
>>> df.columns
Index(['i_tp','i_fp','i_tn','i_fn','j_tp','j_fp','j_tn','j_fn','ir','gr'], ...)
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the parent directory importable so we can re-use sets_creation.py
# ---------------------------------------------------------------------------
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sets_creation import generate_dataset  # noqa: E402 – local import
import utils  # noqa: E402 – local import

# Column names matching the project convention
_CM_COLS = ["i_tp", "i_fp", "i_tn", "i_fn", "j_tp", "j_fp", "j_tn", "j_fn"]


def generate_cm_pairs(n: int = 24) -> pd.DataFrame:
    """Return a DataFrame with every feasible 8-cell confusion-matrix pair
    summing to *n*, augmented with ``ir`` (imbalance ratio) and ``gr``
    (group ratio) columns.

    Parameters
    ----------
    n : int
        Total sample budget (sum of all 8 cells).  The number of rows grows
        combinatorially (~C(n+7,7)); n=24 → ~2 500 rows, n=48 → ~1.2 M rows.
        Values above ~40 may be slow inside Streamlit.

    Returns
    -------
    pd.DataFrame
        Columns: i_tp, i_fp, i_tn, i_fn, j_tp, j_fp, j_tn, j_fn, ir, gr.
        All count columns are int16.  ir and gr are float64 in [0, 1].

    Notes
    -----
    ``generate_dataset(n_cells, k)`` in sets_creation.py uses *n_cells* = 8
    (fixed) and *k* = budget.  Here we hard-code n_cells=8 and expose only
    the budget as the public parameter.
    """
    # generate_dataset prints progress – suppress for library use
    raw: np.ndarray = generate_dataset(n=8, k=n)  # shape (m, 8), dtype int8

    df = pd.DataFrame(raw.astype(np.int16), columns=_CM_COLS)

    # Sanity: every row must sum to n
    assert (df[_CM_COLS].sum(axis=1) == n).all(), (
        "Row sum ≠ n; generation may be corrupt."
    )

    # Augment with IR and GR using the project's vectorised helpers
    df["ir"] = utils.get_imbalance_ratios(df).astype(np.float64)
    df["gr"] = utils.get_group_ratios(df).astype(np.float64)

    return df
