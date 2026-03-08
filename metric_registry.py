"""Metric registry and helpers for the fairness-measure Streamlit app.

The registry is intentionally simple so that new metrics can be added by copying
an existing MetricSpec and calling ``register_metric``. All built-in fairness
metrics follow the sign convention used in the provided repository: values are
computed as ``j - i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd


SeriesLike = pd.Series
MetricFunc = Callable[[pd.DataFrame], SeriesLike]


@dataclass(frozen=True)
class MetricSpec:
    """Description of a metric that can be computed from confusion-matrix rows."""

    key: str
    label: str
    category: str
    description: str
    formula: str
    compute: MetricFunc
    sort_order: int = 999  # lower = earlier in list_metrics; classical metrics use 0–7


_METRICS: Dict[str, MetricSpec] = {}


COUNT_COLUMNS = [
    "i_tp",
    "i_fp",
    "i_tn",
    "i_fn",
    "j_tp",
    "j_fp",
    "j_tn",
    "j_fn",
]


def safe_divide(numerator: SeriesLike | np.ndarray, denominator: SeriesLike | np.ndarray) -> np.ndarray:
    """Return ``numerator / denominator`` with NaN where the denominator is zero."""

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full_like(num, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def require_count_columns(df: pd.DataFrame) -> None:
    missing = [c for c in COUNT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required confusion-matrix columns: {missing}")


def register_metric(spec: MetricSpec) -> None:
    """Register a new metric.

    Existing keys are overwritten on purpose so that users can quickly modify a
    metric by re-registering it in ``custom_metrics.py``.
    """

    _METRICS[spec.key] = spec


def list_metrics(category: str | None = None) -> list[MetricSpec]:
    specs = list(_METRICS.values())
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    return sorted(specs, key=lambda spec: (spec.category, spec.sort_order, spec.label))


def get_metric(key: str) -> MetricSpec:
    if key not in _METRICS:
        raise KeyError(f"Unknown metric key: {key}")
    return _METRICS[key]


def metric_labels(keys: Iterable[str]) -> dict[str, str]:
    return {key: get_metric(key).label for key in keys}


def compute_metric(df: pd.DataFrame, key: str) -> pd.Series:
    require_count_columns(df)
    spec = get_metric(key)
    values = spec.compute(df)
    return pd.Series(np.asarray(values, dtype=np.float64), index=df.index, name=key)


def compute_metrics(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    require_count_columns(df)
    out = pd.DataFrame(index=df.index)
    for key in keys:
        out[key] = compute_metric(df, key)
    return out


def metrics_metadata_frame() -> pd.DataFrame:
    rows = []
    for spec in list_metrics():
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "category": spec.category,
                "formula": spec.formula,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)
