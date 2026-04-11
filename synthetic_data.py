"""Synthetic confusion-matrix generation and dataset helpers."""

from __future__ import annotations

import math
import pickle
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from builtin_metrics import accuracy, g_mean, group_ratio_i, group_ratio_j, imbalance_ratio, stereotypical_ratio, stereotypical_ratio_negative, stereotypical_ratio_combined
from metric_registry import COUNT_COLUMNS, compute_metrics


DEFAULT_MAX_EXACT_ROWS = 20_000_000


def count_confusion_matrices(total: int, parts: int = 8) -> int:
    """Return the number of non-negative integer ``parts``-tuples summing to ``total``."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if parts < 1:
        raise ValueError("parts must be positive")
    return math.comb(total + parts - 1, parts - 1)


def iter_confusion_matrices(total: int, parts: int = 8):
    """Yield all non-negative integer ``parts``-tuples summing to ``total``.

    Uses a stars-and-bars representation, which is easier to reason about than
    the original script and still produces the same set of tuples.
    """

    if total < 0:
        raise ValueError("total must be non-negative")
    if parts < 1:
        raise ValueError("parts must be positive")

    if parts == 1:
        yield (total,)
        return

    for bars in combinations(range(total + parts - 1), parts - 1):
        prev = -1
        row = []
        for bar in bars:
            row.append(bar - prev - 1)
            prev = bar
        row.append(total + parts - 2 - prev)
        yield tuple(row)


def generate_exact_confusion_matrices(
    total: int,
    *,
    parts: int = 8,
    max_rows: int = DEFAULT_MAX_EXACT_ROWS,
) -> pd.DataFrame:
    """Generate every confusion-matrix row exactly.

    This is intended for interactive experimentation on moderate totals. The
    paper-scale case ``n=56`` contains 553,270,671 rows and is not practical in
    a Streamlit session without precomputation.
    """

    rows = count_confusion_matrices(total, parts)
    if rows > max_rows:
        raise ValueError(
            f"Exact generation would create {rows:,} rows, which exceeds the configured limit "
            f"of {max_rows:,}. Use Monte Carlo sampling or reduce n."
        )

    dtype = np.int16 if total <= np.iinfo(np.int16).max else np.int32
    data = np.empty((rows, parts), dtype=dtype)
    for idx, row in enumerate(iter_confusion_matrices(total, parts)):
        data[idx] = row

    return pd.DataFrame(data, columns=COUNT_COLUMNS)


def _bars_to_counts(total: int, parts: int, bars: np.ndarray) -> np.ndarray:
    prev = -1
    counts: list[int] = []
    for bar in bars:
        counts.append(int(bar - prev - 1))
        prev = int(bar)
    counts.append(total + parts - 2 - prev)
    return np.asarray(counts, dtype=np.int32)


def sample_uniform_confusion_matrices(
    total: int,
    draws: int,
    *,
    parts: int = 8,
    seed: int = 2137,
) -> pd.DataFrame:
    """Sample confusion matrices uniformly from the space of all integer compositions.

    Each sampled row is an unbiased draw from the set of all non-negative
    ``parts``-tuples summing to ``total``.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")

    rng = np.random.default_rng(seed)
    data = np.empty((draws, parts), dtype=np.int32)
    max_bar = total + parts - 1
    for idx in range(draws):
        bars = np.sort(rng.choice(max_bar, size=parts - 1, replace=False))
        data[idx] = _bars_to_counts(total, parts, bars)
    return pd.DataFrame(data, columns=COUNT_COLUMNS)


def load_confusion_matrices_from_pickle(path_or_bytes: str | Path | bytes) -> pd.DataFrame:
    """Load confusion matrices saved by the original repository or this app."""

    if isinstance(path_or_bytes, (str, Path)):
        with open(path_or_bytes, "rb") as handle:
            obj = pickle.load(handle)
    else:
        obj = pickle.loads(path_or_bytes)
    df = pd.DataFrame(obj, columns=COUNT_COLUMNS)
    return df


def dump_confusion_matrices_to_pickle(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df[COUNT_COLUMNS].to_numpy())


def ratio_values(total: int, *, min_count: int = 0) -> list[float]:
    """Return all attainable ratios k / total for k in [min_count, total]."""

    return [k / total for k in range(min_count, total + 1)]


def paper_ratio_defaults(total: int) -> list[float]:
    """Return the paper-style five-ratio panel values for a given total.

    For the high-imbalance edge values, the function uses ``2 / total`` so that
    the minority side can still be represented in both groups/classes.
    """

    if total < 4:
        return sorted(set(ratio_values(total)))
    low = max(2 / total, 1 / total)
    return [low, 0.25, 0.5, 0.75, 1.0 - low]


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with frequently used derived columns added."""

    out = df.copy()
    out["total"] = out[COUNT_COLUMNS].sum(axis=1).astype(np.int32)
    out["positive_total"] = (out["i_tp"] + out["i_fn"] + out["j_tp"] + out["j_fn"]).astype(np.int32)
    out["negative_total"] = (out["i_fp"] + out["i_tn"] + out["j_fp"] + out["j_tn"]).astype(np.int32)
    out["i_total"] = (out["i_tp"] + out["i_fp"] + out["i_tn"] + out["i_fn"]).astype(np.int32)
    out["j_total"] = (out["j_tp"] + out["j_fp"] + out["j_tn"] + out["j_fn"]).astype(np.int32)
    out["group_ratio_j"] = np.asarray(group_ratio_j(out), dtype=np.float32)
    out["group_ratio_i"] = np.asarray(group_ratio_i(out), dtype=np.float32)
    out["imbalance_ratio"] = np.asarray(imbalance_ratio(out), dtype=np.float32)
    out["stereotypical_ratio"] = np.asarray(stereotypical_ratio(out), dtype=np.float32)
    out["stereotypical_ratio_negative"] = np.asarray(stereotypical_ratio_negative(out), dtype=np.float32)
    out["stereotypical_ratio_combined"] = np.asarray(stereotypical_ratio_combined(out), dtype=np.float32)
    out["accuracy"] = np.asarray(accuracy(out), dtype=np.float32)
    out["g_mean"] = np.asarray(g_mean(out), dtype=np.float32)
    return out


def with_selected_metrics(df: pd.DataFrame, metric_keys: Iterable[str]) -> pd.DataFrame:
    out = add_base_columns(df)
    metric_keys = list(metric_keys)
    if metric_keys:
        metric_frame = compute_metrics(out, metric_keys).astype(np.float32)
        for column in metric_frame.columns:
            out[column] = metric_frame[column]
    return out
