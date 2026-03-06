"""
generate.py
-----------
Generates or loads all feasible confusion-matrix pairs whose 8 cell counts
sum exactly to a given budget *n*.

Public API
----------
generate_cm_pairs(n)          – enumerate from scratch (calls sets_creation.py)
load_cm_pairs(path)           – load a pre-generated .bin file
detect_n_from_array(X)        – infer n from a loaded raw array
"""

from __future__ import annotations

import pickle
import sys
import os

import numpy as np
import pandas as pd

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sets_creation import generate_dataset  # noqa: E402
import utils                                # noqa: E402

_CM_COLS = ["i_tp", "i_fp", "i_tn", "i_fn", "j_tp", "j_fp", "j_tn", "j_fn"]


def _build_df(raw: np.ndarray) -> pd.DataFrame:
    """Convert a raw (m, 8) int array to an annotated DataFrame."""
    df = pd.DataFrame(raw.astype(np.int16), columns=_CM_COLS)
    df["ir"] = utils.get_imbalance_ratios(df).astype(np.float64)
    df["gr"] = utils.get_group_ratios(df).astype(np.float64)
    return df


def detect_n_from_array(X: np.ndarray) -> int:
    """Return the row sum (= n) from a raw simplex array."""
    return int(X[0].sum())


def generate_cm_pairs(n: int = 24) -> pd.DataFrame:
    """Return a DataFrame with every feasible 8-cell CM pair summing to *n*."""
    raw: np.ndarray = generate_dataset(n=8, k=n)
    return _build_df(raw.astype(np.int16))


def load_cm_pairs(path: str) -> pd.DataFrame:
    """Load a pre-generated .bin file and return an annotated DataFrame.

    The .bin file is a pickle of the raw (m, 8) int array produced by
    sets_creation.py  (e.g. ``Set(08,56).bin`` where 56 is k=n).
    n is auto-detected from the row sum.
    """
    with open(path, "rb") as f:
        X = pickle.load(f)
    X = np.asarray(X, dtype=np.int16)
    if X.ndim != 2 or X.shape[1] != 8:
        raise ValueError(
            f"Expected a (m, 8) array in {path!r}; got shape {X.shape}."
        )
    return _build_df(X)
